"""Three ways to find where a capture was taken, all trained only on renders of the CAD model.

retrieval   global image descriptor (HOG) -> pose of the most similar reference render
features    SIFT keypoints matched to keypoints of all renders, whose pixels have known 3D points -> PnP + RANSAC
forest      ML: a random forest predicts the 3D surface point of every pixel (scene coordinate regression,
            after Shotton et al. 2013) -> PnP + RANSAC
hybrid      features + forest correspondences in one RANSAC
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors

from .render import Camera, Capture

FG_THRESHOLD = 24          # grey level above which a pixel is treated as part, not background


@dataclass
class Estimate:
    R: np.ndarray | None
    t: np.ndarray | None
    inliers: int = 0           # confidence: RANSAC inliers (or descriptor similarity x 100 for retrieval)
    residual_px: float = float("nan")   # CAD edge fit (cnn_fit only): median edge distance, px; lower is better
    agreement: float = float("nan")     # CAD edge fit (cnn_fit only): share of CAD edges found in the image


def _descriptor(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    d = hog(small, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return d / (np.linalg.norm(d) + 1e-9)


def foreground(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (5, 5), 0) > FG_THRESHOLD


def _pnp(pts2d: np.ndarray, pts3d: np.ndarray, K: np.ndarray, reproj_px: float = 4.0, iters: int = 1000) -> Estimate:
    if len(pts2d) < 6:
        return Estimate(None, None, 0)
    cv2.setRNGSeed(0)                       # RANSAC is random: fix it so runs are repeatable
    ok, rvec, tvec, inl = cv2.solvePnPRansac(pts3d.astype(np.float64), pts2d.astype(np.float64), K, None,
                                             reprojectionError=reproj_px, iterationsCount=iters,
                                             confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 6:
        return Estimate(None, None, 0 if inl is None else len(inl))
    idx = np.asarray(inl)[:, 0].astype(int)
    rvec, tvec = cv2.solvePnPRefineLM(pts3d[idx].astype(np.float64), pts2d[idx].astype(np.float64), K, None, rvec, tvec)
    return Estimate(cv2.Rodrigues(rvec)[0], tvec[:, 0], len(idx))


class Retrieval:
    name = "retrieval"

    def __init__(self, refs: list[Capture]):
        self.refs = refs
        self.desc = np.array([_descriptor(r.image) for r in refs])
        self.nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(self.desc)

    def topk(self, img: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        dist, idx = self.nn.kneighbors(_descriptor(img)[None], n_neighbors=k)
        return idx[0], 1 - dist[0]

    def locate(self, img: np.ndarray) -> Estimate:
        idx, sim = self.topk(img, 1)
        r = self.refs[idx[0]]
        return Estimate(r.R, r.t, int(100 * sim[0]))


class Features:
    """SIFT keypoints of every reference render, each with its 3D point. A query keypoint is matched to the whole
    set; the ratio test compares the best match with the best one that lies elsewhere on the part, because the same
    3D point seen in many renders would otherwise fail a plain ratio test."""
    name = "features"

    def __init__(self, refs: list[Capture], camera: Camera, ratio: float = 0.8, min_sep_mm: float = 8.0):
        self.K, self.ratio, self.min_sep = camera.K, ratio, min_sep_mm
        self.sift = cv2.SIFT_create(nfeatures=300, contrastThreshold=0.02)  # pyright: ignore[reportAttributeAccessIssue]
        p3s, dess = [], []
        for r in refs:
            kps, des = self.sift.detectAndCompute(r.image, None)
            if des is None:
                continue
            uv = np.array([k.pt for k in kps])
            p3 = r.xyz[np.clip(uv[:, 1].astype(int), 0, r.image.shape[0] - 1),
                       np.clip(uv[:, 0].astype(int), 0, r.image.shape[1] - 1)]
            ok = np.isfinite(p3).all(1)
            p3s.append(p3[ok])
            dess.append(des[ok])
        self.p3 = np.vstack(p3s)
        self.index = NearestNeighbors(n_neighbors=12).fit(np.vstack(dess))

    def correspondences(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kps, des = self.sift.detectAndCompute(img, None)
        if des is None or len(kps) < 6:
            return np.zeros((0, 2)), np.zeros((0, 3))
        uv = np.array([k.pt for k in kps])
        dist, idx = self.index.kneighbors(des)
        p2, p3 = [], []
        for q in range(len(uv)):
            best = self.p3[idx[q, 0]]
            far = np.linalg.norm(self.p3[idx[q, 1:]] - best, axis=1) > self.min_sep
            second = dist[q, 1:][far][0] if far.any() else np.inf
            if dist[q, 0] < self.ratio * second:
                p2.append(uv[q])
                p3.append(best)
        return np.array(p2).reshape(-1, 2), np.array(p3).reshape(-1, 3)

    def locate(self, img: np.ndarray) -> Estimate:
        return _pnp(*self.correspondences(img), self.K)


def _rings(win: int) -> np.ndarray:
    r = win // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    d = np.sqrt(xx ** 2 + yy ** 2)
    return np.stack([(d >= i - 0.5) & (d < i + 0.5) for i in range(r + 1)]).reshape(r + 1, -1).astype(np.float32)


RINGS = _rings(11)


def pixel_features(img: np.ndarray, uv: np.ndarray, win: int = 11, oriented: bool = True) -> np.ndarray:
    """Description of the neighbourhood of each pixel at three scales (11, 22 and 44 px of context), normalised for
    contrast: the mean and spread of intensity on rings of growing radius, which do not depend on camera roll, and,
    when roll is roughly known (oriented=True), the 7 x 7 patch itself."""
    out = []
    level = cv2.GaussianBlur(img.astype(np.float32) / 255, (0, 0), 0.8)
    r = win // 2
    ring_n = RINGS.sum(1)
    for s in range(3):
        pad = np.pad(level, r, mode="edge")
        win_view = np.lib.stride_tricks.sliding_window_view(pad, (win, win))
        x = np.clip((uv[:, 0] / 2 ** s).astype(int), 0, level.shape[1] - 1)
        y = np.clip((uv[:, 1] / 2 ** s).astype(int), 0, level.shape[0] - 1)
        p = win_view[y, x].reshape(len(uv), -1)
        m, sd = p.mean(1, keepdims=True), p.std(1, keepdims=True)
        pn = (p - m) / (sd + 0.03)
        ring_mean = pn @ RINGS.T / ring_n
        ring_sq = (pn ** 2) @ RINGS.T / ring_n
        ring_sd = np.sqrt(np.clip(ring_sq - ring_mean ** 2, 0, None))
        out += [ring_mean, ring_sd, m, sd]
        if oriented:
            out.append(pn.reshape(len(uv), win, win)[:, 2:-2:1, 2:-2:1].reshape(len(uv), -1))
        level = cv2.pyrDown(level)
    return np.hstack(out).astype(np.float32)


class Forest:
    name = "forest"

    def __init__(self, refs: list[Capture], camera: Camera, size_mm: float, pixels_per_image: int = 150,
                 n_trees: int = 32, seed: int = 0, stride: int = 4, oriented: bool = True):
        self.K, self.stride, self.scale, self.oriented = camera.K, stride, size_mm, oriented
        rng = np.random.default_rng(seed)
        X, Y = [], []
        for r in refs:
            ys, xs = np.nonzero(r.mask)
            pick = rng.choice(len(xs), min(pixels_per_image, len(xs)), replace=False)
            uv = np.stack([xs[pick], ys[pick]], 1) + 0.5
            X.append(pixel_features(r.image, uv, oriented=oriented))
            Y.append(r.xyz[ys[pick], xs[pick]] / size_mm)
        self.model = ExtraTreesRegressor(n_estimators=n_trees, min_samples_leaf=3, max_features=0.33,
                                         n_jobs=-1, random_state=seed).fit(np.vstack(X), np.vstack(Y))
        self.n_samples = int(sum(len(x) for x in X))

    def correspondences(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fg = foreground(img)
        ys, xs = np.nonzero(fg[::self.stride, ::self.stride])
        if len(xs) < 6:
            return np.zeros((0, 2)), np.zeros((0, 3))
        uv = np.stack([xs, ys], 1) * self.stride + 0.5
        return uv, self.model.predict(pixel_features(img, uv, oriented=self.oriented)) * self.scale

    def locate(self, img: np.ndarray) -> Estimate:
        return _pnp(*self.correspondences(img), self.K, reproj_px=5.0, iters=2000)


class Hybrid:
    name = "hybrid"

    def __init__(self, features: Features, forest: Forest, camera: Camera):
        self.f, self.g, self.K = features, forest, camera.K

    def locate(self, img: np.ndarray) -> Estimate:
        a2, a3 = self.f.correspondences(img)
        b2, b3 = self.g.correspondences(img)
        return _pnp(np.vstack([a2, b2]), np.vstack([a3, b3]), self.K, reproj_px=5.0, iters=2000)
