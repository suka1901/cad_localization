"""Fit the camera pose to the image with the CAD model (render-and-compare), after the CNN has proposed one.

The CNN + PnP gives a first pose. This step then adjusts the six pose parameters (three for the camera position in
mm, three for its orientation) so that the visible edges of the CAD model, projected with that pose, fall on the
edges in the image. It is a small, physical model with a clear meaning, fitted to the data by non-linear least
squares, in the same spirit as model-based tracking (Drummond and Cipolla, PAMI 2002) and as fitting a dynamic model
to logged data in system identification.

What it gives:
  - a refined pose,
  - the fit residual: how far, in pixels, the CAD edges lie from the image edges at the fitted pose. It is a
    physical check that does not depend on the CNN, so it shows when the CNN's answer is wrong.

Edges of the CAD model are of two kinds: sharp edges (fixed, where faces meet at more than `sharp_deg`) and
silhouette edges of curved surfaces (they depend on the view: where the surface turns away from the camera).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

from .methods import Estimate, _pnp
from .render import Renderer


@dataclass
class Fit:
    R: np.ndarray
    t: np.ndarray
    residual_px: float          # median distance of the visible CAD edges to the nearest image edge, at the fit
    agreement: float            # share of the visible CAD edge points that lie within 2 px of an image edge
    edge_points: int            # visible CAD edge points used
    start_residual_px: float    # the same two numbers at the starting pose (from the CNN)
    start_agreement: float
    improved: bool              # False: the fit did not improve the agreement, so the starting pose is kept


def _edge_samples(a: np.ndarray, b: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every `step` mm along segments a-b; returns the points and the index of their segment."""
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / step).astype(int))
    seg = np.repeat(np.arange(len(a)), n)
    s = (np.concatenate([np.arange(k) for k in n]) + 0.5) / np.repeat(n, n)
    return a[seg] + s[:, None] * (b[seg] - a[seg]), seg


class EdgeModel:
    """The edges of the CAD model, and how to project the visible ones for a pose."""

    def __init__(self, renderer: Renderer, sharp_deg: float = 25.0, step_mm: float = 1.5):
        self.r = renderer
        m = renderer.part.mesh
        ang = m.face_adjacency_angles
        edges = m.vertices[m.face_adjacency_edges]                 # (E, 2, 3)
        sharp = ang > np.radians(sharp_deg)
        smooth = (ang > np.radians(1.0)) & ~sharp
        self.sharp_pts, _ = _edge_samples(edges[sharp, 0], edges[sharp, 1], step_mm)
        self.smooth_pts, seg = _edge_samples(edges[smooth, 0], edges[smooth, 1], step_mm)
        faces = m.face_adjacency[smooth]                           # the two faces on each smooth edge
        self.smooth_n = m.face_normals[faces[seg]]                 # (P, 2, 3)

    def visible(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Edge points (part frame) that the camera sees at this pose: sharp edges, plus silhouettes of curved
        surfaces, minus those hidden behind other parts of the part or outside the image."""
        c = -R.T @ t
        v = c - self.smooth_pts
        s = np.einsum("pkj,pj->pk", self.smooth_n, v)
        silhouette = (s[:, 0] > 0) != (s[:, 1] > 0)
        pts = np.vstack([self.sharp_pts, self.smooth_pts[silhouette]])
        cam = pts @ R.T + t
        K = self.r.cam.K
        z = cam[:, 2]
        uv = cam[:, :2] / np.maximum(z, 1e-6)[:, None] * K[0, 0] + K[:2, 2]
        w, h = self.r.cam.width, self.r.cam.height
        inside = (z > 1) & (uv[:, 0] > 2) & (uv[:, 0] < w - 2) & (uv[:, 1] > 2) & (uv[:, 1] < h - 2)
        pts = pts[inside]
        if len(pts) == 0:
            return pts
        d = pts - c
        dist = np.linalg.norm(d, axis=1)
        rays = np.hstack([np.broadcast_to(c, d.shape), d / dist[:, None]]).astype(np.float32)
        hit = self.r.scene.cast_rays(o3d.core.Tensor.from_numpy(np.ascontiguousarray(rays)))["t_hit"].numpy()
        return pts[hit > dist - 1.0]                               # nothing in front of the edge point


def edge_distance(img: np.ndarray, blur: float = 1.2, low: int = 20, high: int = 60) -> np.ndarray:
    """Distance (px) from every pixel to the nearest image edge."""
    e = cv2.Canny(cv2.GaussianBlur(img, (0, 0), blur), low, high)
    return cv2.distanceTransform((e == 0).astype(np.uint8), cv2.DIST_L2, 3)


def _sample(dt: np.ndarray, uv: np.ndarray, cap: float) -> np.ndarray:
    h, w = dt.shape
    out = np.full(len(uv), cap, dtype=float)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    x, y = uv[ok, 0], uv[ok, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    v = (dt[y0, x0] * (1 - fx) * (1 - fy) + dt[y0, x0 + 1] * fx * (1 - fy)
         + dt[y0 + 1, x0] * (1 - fx) * fy + dt[y0 + 1, x0 + 1] * fx * fy)
    out[ok] = np.minimum(v, cap)
    return out


def _project(K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, pts: np.ndarray) -> np.ndarray:
    R = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))[0]
    c = pts @ R.T + np.asarray(tvec, dtype=float).reshape(3)
    return c[:, :2] / np.maximum(c[:, 2:], 1e-6) * K[0, 0] + K[:2, 2]


def fit_pose(model: EdgeModel, img: np.ndarray, R: np.ndarray, t: np.ndarray,
             anchor: tuple[np.ndarray, np.ndarray] | None = None, anchor_weight: float = 0.3,
             stages: tuple[float, ...] = (20.0, 8.0, 4.0), rounds: int = 2) -> Fit:
    """Least-squares fit of the six pose parameters to the image edges, from the starting pose (R, t).

    Residuals: for every visible CAD edge point, the distance (px) from its projection to the nearest image edge,
    truncated at `stage` px (coarse first, then fine), with a robust loss so that edges missing in the image do not
    dominate. `anchor` (2D, 3D points: the CNN's RANSAC inliers) keeps the fit close to what the CNN saw; it is
    weighted lightly, so the CAD edges decide.
    """
    K = model.r.cam.K
    dt = edge_distance(img)
    rvec = cv2.Rodrigues(R)[0][:, 0]
    tvec = t.astype(float).copy()
    start = _score(model, dt, R, t, stages[-1])
    for cap in stages:
        for _ in range(rounds):
            pts = model.visible(cv2.Rodrigues(rvec)[0], tvec)
            if len(pts) < 20:
                break

            def res(x: np.ndarray) -> np.ndarray:
                r = _sample(dt, _project(K, x[:3], x[3:], pts), cap) / np.sqrt(len(pts))
                if anchor is not None and len(anchor[0]):
                    d = _project(K, x[:3], x[3:], anchor[1]) - anchor[0]
                    a = np.minimum(np.linalg.norm(d, axis=1), cap) * anchor_weight / np.sqrt(len(d))
                    r = np.concatenate([r, a])
                return r

            sol = least_squares(res, np.concatenate([rvec, tvec]), loss="soft_l1", f_scale=2.0 / np.sqrt(len(pts)),
                                x_scale=np.array([0.02, 0.02, 0.02, 2.0, 2.0, 2.0]), max_nfev=60)
            rvec, tvec = sol.x[:3], sol.x[3:]
    R1 = cv2.Rodrigues(rvec)[0]
    end = _score(model, dt, R1, tvec, stages[-1])
    if end[1] < start[1]:                  # the fit made the CAD edges agree less with the image: keep the start
        return Fit(R, t, start[0], start[1], start[2], start[0], start[1], False)
    return Fit(R1, tvec, end[0], end[1], end[2], start[0], start[1], True)


def _score(model: EdgeModel, dt: np.ndarray, R: np.ndarray, t: np.ndarray, cap: float) -> tuple[float, float, int]:
    """Median distance (px) of the visible CAD edges to the image edges, and the share within 2 px."""
    pts = model.visible(R, t)
    if len(pts) == 0:
        return np.inf, 0.0, 0
    d = _sample(dt, _project(model.r.cam.K, cv2.Rodrigues(R)[0], t.astype(float), pts), cap)
    return float(np.median(d)), float(np.mean(d <= 2.0)), len(pts)


class CNNFit:
    """ML proposes, geometry checks: the CNN + PnP gives up to `hypotheses` candidate poses (RANSAC is run again on
    the cells the earlier candidates did not explain), the CAD edge fit refines each, and the candidate whose CAD
    edges agree best with the image wins. This targets the case where wrong cells agree on a look-alike place."""
    name = "cnn_fit"

    def __init__(self, cnn, renderer: Renderer, anchor_weight: float = 0.3,
                 stages: tuple[float, ...] = (20.0, 8.0, 4.0), hypotheses: int = 3, min_inliers: int = 12):
        self.cnn, self.edges, self.anchor_weight, self.stages = cnn, EdgeModel(renderer), anchor_weight, stages
        self.hypotheses, self.min_inliers = hypotheses, min_inliers
        self.K = renderer.cam.K
        self.last: list[tuple[Estimate, Fit]] = []          # the candidates of the last call, for inspection

    def candidates(self, p2: np.ndarray, p3: np.ndarray) -> list[tuple[Estimate, np.ndarray]]:
        out, left = [], np.ones(len(p2), dtype=bool)
        for _ in range(self.hypotheses):
            if left.sum() < self.min_inliers:
                break
            est = _pnp(p2[left], p3[left], self.K, reproj_px=6.0, iters=1000)
            if est.R is None or est.t is None or est.inliers < self.min_inliers:
                break
            proj = cv2.projectPoints(p3.astype(np.float64), cv2.Rodrigues(est.R)[0], est.t.astype(np.float64),
                                     self.K, None)[0][:, 0]
            inl = np.linalg.norm(proj - p2, axis=1) < 6.0
            out.append((est, inl))
            left &= ~inl
        return out

    def locate(self, img: np.ndarray) -> Estimate:
        p2, p3 = self.cnn.correspondences(img)
        cands = self.candidates(p2, p3)
        self.last = []
        if not cands:
            return _pnp(p2, p3, self.K, reproj_px=6.0, iters=1000)
        for est, inl in cands:
            assert est.R is not None and est.t is not None
            f = fit_pose(self.edges, img, est.R, est.t, (p2[inl], p3[inl]), self.anchor_weight, self.stages)
            self.last.append((est, f))
        est, f = max(self.last, key=lambda ef: ef[1].agreement)
        return Estimate(f.R, f.t, est.inliers, residual_px=f.residual_px, agreement=f.agreement)


def overlay(model: EdgeModel, img: np.ndarray, poses: list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]],
            scale: int = 2) -> np.ndarray:
    """The image with the visible CAD edges drawn for each pose (BGR colour per pose): what the fit looks at."""
    big = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    big = (big * 0.75).astype(np.uint8)
    for R, t, color in poses:
        pts = model.visible(R, t)
        if len(pts) == 0:
            continue
        uv = _project(model.r.cam.K, cv2.Rodrigues(R)[0], t.astype(float), pts) * scale
        for u, v in uv.astype(int):
            cv2.circle(big, (int(u), int(v)), 1, color, -1)
    return big
