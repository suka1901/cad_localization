"""Where on the part will localisation be hard? Predicted from the CAD model alone.

For a view of the part, four numbers are computed from the CAD model only (no image, no network):

  edges         share of the visible part that lies on a CAD edge (corners, hole rims, rib edges): structure to
                recognise
  orientations  how many different surface orientations are in view (entropy of the face normals, in bits)
  coverage      share of the image that shows the part
  look_alike    how much more the plain CAD view resembles a view of a different place on the part (more than 30 mm
                away) than a view of the same place (HOG similarity of the best match of each, difference): above 0
                means a look-alike place exists

A logistic model turns them into the probability that a capture from that view is located. Its coefficients say how
much each property matters, and the map over the surface shows where captures will be hard before any capture is
taken, much as an identifiability analysis says which parameters of a model the data can pin down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from .calibration import Logistic, auc
from .methods import Retrieval
from .render import Capture, Renderer, sample_capture

FEATURES = ["edges", "orientations", "coverage", "look_alike"]
FEATURE_TEXT = {
    "edges": "CAD edges in view (share of the part)",
    "orientations": "surface orientations in view (bits)",
    "coverage": "share of the image showing the part",
    "look_alike": "look-alike margin: other place vs same place",
}


def view_features(renderer: Renderer, R: np.ndarray, t: np.ndarray, retrieval: Retrieval,
                  sharp_deg: float = 25.0, min_sep_mm: float = 30.0) -> dict[str, float]:
    """The four CAD-only numbers for the view from pose (R, t)."""
    cam = renderer.cam
    pts, prim, hit, _ = renderer.raycast(R, t)
    h, w = cam.height, cam.width
    hit2 = hit.reshape(h, w)
    n = renderer.part.mesh.face_normals[np.where(hit, prim, 0)].reshape(h, w, 3)
    p = np.nan_to_num(pts.reshape(h, w, 3))
    edge = np.zeros((h, w), dtype=bool)
    cos = np.cos(np.radians(sharp_deg))
    for dy, dx in ((0, 1), (1, 0)):
        a, b = (slice(0, h - dy), slice(0, w - dx)), (slice(dy, h), slice(dx, w))
        both = hit2[a] & hit2[b]
        crease = both & ((n[a] * n[b]).sum(-1) < cos)
        step = both & (np.linalg.norm(p[a] - p[b], axis=-1) > 3.0)       # depth step: an occluding edge
        border = hit2[a] != hit2[b]                                     # part outline against the background
        e = crease | step | border
        edge[a] |= e
        edge[b] |= e
    cover = float(hit2.mean())
    edges = float((edge & hit2).sum() / max(hit2.sum(), 1))
    # orientations: normals binned on a coarse sphere grid, entropy weighted by visible area
    nn = n[hit2]
    if len(nn):
        az = np.floor((np.arctan2(nn[:, 1], nn[:, 0]) + np.pi) / (np.pi / 4)).astype(int)
        el = np.floor((np.arcsin(np.clip(nn[:, 2], -1, 1)) + np.pi / 2) / (np.pi / 4)).astype(int)
        _, counts = np.unique(az * 10 + el, return_counts=True)
        q = counts / counts.sum()
        orient = float(-(q * np.log2(q)).sum())
    else:
        orient = 0.0
    # look-alike: the plain CAD render of this view against the CAD reference renders of other places
    img = renderer.render(R, t, np.array([0.0, 0.0, -1.0]), np.random.default_rng(0), noise=0.0).image
    idx, sim = retrieval.topk(img, min(200, len(retrieval.refs)))
    i = (h // 2) * w + w // 2
    centre = pts[i] if hit[i] else np.nan_to_num(pts[hit].mean(0)) if hit.any() else np.zeros(3)
    d = np.array([np.linalg.norm(retrieval.refs[j].target - centre) for j in idx])
    s_far = sim[d > min_sep_mm][0] if (d > min_sep_mm).any() else sim[-1]
    s_near = sim[d <= min_sep_mm][0] if (d <= min_sep_mm).any() else sim[-1]
    return {"edges": edges, "orientations": orient, "coverage": cover, "look_alike": float(s_far - s_near)}


def features_for(renderer: Renderer, caps: list[Capture], retrieval: Retrieval) -> pd.DataFrame:
    """CAD-only features of each capture's view (at its true pose; in use, at the estimated pose)."""
    return pd.DataFrame([view_features(renderer, c.R, c.t, retrieval) for c in caps])


class Localisability:
    """Logistic model: CAD-only view features -> probability that a capture is located."""

    def __init__(self):
        self.lr = Logistic()

    def fit(self, X: pd.DataFrame, located: np.ndarray) -> "Localisability":
        self.lr.fit(X[FEATURES].to_numpy(), located)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr.predict(X[FEATURES].to_numpy())

    def effects(self) -> pd.DataFrame:
        """Standardised coefficients: change in log-odds of success for one standard deviation of the feature."""
        return pd.DataFrame({"feature": [FEATURE_TEXT[f] for f in FEATURES],
                             "effect_per_sd": self.lr.coef(), "feature_sd": self.lr.scaler.scale_})


def single_feature_auc(X: pd.DataFrame, located: np.ndarray) -> dict[str, float]:
    """How well each CAD-only number alone separates located from missed captures (0.5 = not at all)."""
    out = {}
    for f in FEATURES:
        a = auc(located, X[f].to_numpy())
        out[f] = float(max(a, 1 - a))
    return out


def surface_map(renderer: Renderer, model: Localisability, retrieval: Retrieval, n: int, seed: int,
                dist=(60, 180), max_roll_deg: float | None = 20) -> pd.DataFrame:
    """Predicted chance of locating a normal inspection photo of each spot on the surface (CAD only)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        cap = sample_capture(renderer, rng, dist, max_roll_deg=max_roll_deg)
        f = view_features(renderer, cap.R, cap.t, retrieval)
        rows.append({"x": cap.target[0], "y": cap.target[1], "z": cap.target[2], "region": cap.target_region, **f})
    df = pd.DataFrame(rows)
    df["p_located"] = model.predict(df)
    return df


def face_values(renderer: Renderer, spots: pd.DataFrame, k: int = 5) -> np.ndarray:
    """Spread the per-spot prediction over the mesh faces (mean of the k nearest spots), for colouring the CAD."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(k, len(spots))).fit(spots[["x", "y", "z"]].to_numpy())
    _, idx = nn.kneighbors(renderer.part.mesh.triangles_center)
    return spots["p_located"].to_numpy()[idx].mean(1)
