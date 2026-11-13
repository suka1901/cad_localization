"""Error budget: which stage of the pipeline caused each miss?

The pipeline has three stages, and each is tested on its own, as one checks each force law of a dynamic model
separately before trusting the full simulation:

  1. CNN: for each image cell, which CAD point does it show? Checked cell by cell against the true point, and split
     into the two parts of the prediction: the surface patch (classification) and the offset inside it (regression).
  2. RANSAC: does it pick the correct cells? Checked by running PnP on the correct cells only.
  3. PnP + geometry: is the pose well determined by the cells at all? Checked by running PnP on the true 3D points
     of the same cells (an oracle CNN).

Every capture that is not located within the tolerance gets one cause:

  too_few_correct   the CNN got too few cells right: even the correct cells alone do not give the place.
                    The network did not recognise this view.
  wrong_consensus   enough correct cells, but wrong cells agreed on a different pose and RANSAC chose it:
                    a look-alike place on the part (ambiguity).
  imprecise         RANSAC chose mostly correct cells, yet the result is off: the cells are too inaccurate or too
                    close together to fix the pose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import centre_point
from .methods import _pnp
from .render import Capture, Renderer

CAUSES = ["located", "too_few_correct", "wrong_consensus", "imprecise", "no_pose"]
CAUSE_TEXT = {
    "located": "located",
    "too_few_correct": "CNN: too few correct cells (view not recognised)",
    "wrong_consensus": "RANSAC: wrong cells agreed on a look-alike place",
    "imprecise": "pose imprecise although the cells are mostly right",
    "no_pose": "no pose found",
}


def _centre_error(renderer: Renderer, cap: Capture, uv: np.ndarray, xyz: np.ndarray) -> float:
    if len(uv) < 6:
        return np.inf
    est = _pnp(uv, xyz, renderer.cam.K, reproj_px=6.0, iters=1000)
    if est.R is None or est.t is None:
        return np.inf
    p, _ = centre_point(renderer, est.R, est.t)
    return np.inf if p is None else float(np.linalg.norm(p - cap.target))


def true_points(cap: Capture, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The CAD point really seen at each pixel, and whether it is the part and not hidden by clutter."""
    h, w = cap.xyz.shape[:2]
    x = np.clip(uv[:, 0].astype(int), 0, w - 1)
    y = np.clip(uv[:, 1].astype(int), 0, h - 1)
    xyz = cap.xyz[y, x].astype(float)
    seen = np.isfinite(xyz).all(1)
    if cap.hidden is not None:
        seen &= ~cap.hidden[y, x]
    return xyz, seen


def diagnose(cnn, renderer: Renderer, cap: Capture, tol_mm: float = 10.0, cell_mm: float = 10.0) -> dict:
    """Test each stage on one capture. `cell_mm`: a cell counts as correct if its 3D point is within this."""
    c = cnn.cells(cap.image)
    fg = c["part"]
    uv = c["uv"][fg]
    pred = cnn.points(c, 0)[fg]
    true, seen = true_points(cap, uv)
    err = np.where(seen, np.linalg.norm(np.nan_to_num(pred - true), axis=1), np.inf)
    good = err <= cell_mm
    true_patch = np.full(len(uv), -1)
    if seen.any():
        true_patch[seen] = np.linalg.norm(true[seen, None] - cnn.centers[None], axis=2).argmin(1)
    patch_ok = seen & (c["patches"][fg, 0] == true_patch)

    # the full pipeline (top-2 candidates per cell, as in CNN.locate), and which cells RANSAC kept
    p2, p3 = cnn.correspondences(cap.image)
    est = _pnp(p2, p3, renderer.cam.K, reproj_px=6.0, iters=1000)
    full_err, inlier_correct = np.inf, np.nan
    if est.R is not None and est.t is not None:
        p, _ = centre_point(renderer, est.R, est.t)
        full_err = np.inf if p is None else float(np.linalg.norm(p - cap.target))
        cam_pts = p3 @ est.R.T + est.t
        proj = cam_pts[:, :2] / np.maximum(cam_pts[:, 2:], 1e-6) * renderer.cam.K[0, 0] + renderer.cam.K[:2, 2]
        inl = np.linalg.norm(proj - p2, axis=1) < 6.0
        t2, s2 = true_points(cap, p2)
        ok2 = s2 & (np.linalg.norm(np.nan_to_num(p3 - t2), axis=1) <= cell_mm)
        inlier_correct = float(ok2[inl].mean()) if inl.any() else np.nan

    oracle_err = _centre_error(renderer, cap, uv[seen], true[seen])
    good_only_err = _centre_error(renderer, cap, uv[good], pred[good])

    if full_err <= tol_mm:
        cause = "located"
    elif not np.isfinite(full_err) and est.R is None:
        cause = "no_pose"
    elif good.sum() < 6 or good_only_err > tol_mm:
        cause = "too_few_correct"
    elif inlier_correct < 0.5:
        cause = "wrong_consensus"
    else:
        cause = "imprecise"
    return {
        "cells_part": int(fg.sum()), "cells_seen": int(seen.sum()), "cells_correct": int(good.sum()),
        "share_correct": float(good.sum() / max(seen.sum(), 1)),
        "patch_accuracy": float(patch_ok.sum() / max(seen.sum(), 1)),
        "offset_error_mm": float(np.median(err[patch_ok])) if patch_ok.any() else np.nan,
        "cell_error_mm": float(np.median(err[seen])) if seen.any() else np.nan,
        "inliers_correct": inlier_correct,
        "error_full_mm": full_err, "error_correct_cells_only_mm": good_only_err, "error_oracle_cnn_mm": oracle_err,
        "cause": cause,
    }


def error_budget(cnn, renderer: Renderer, tests: dict[str, list[Capture]], tol_mm: float = 10.0) -> pd.DataFrame:
    rows = []
    for cond, caps in tests.items():
        for i, cap in enumerate(caps):
            rows.append({"condition": cond, "capture": i, **diagnose(cnn, renderer, cap, tol_mm)})
    return pd.DataFrame(rows)


def budget_table(eb: pd.DataFrame, tol_mm: float = 10.0) -> pd.DataFrame:
    """Per condition: how good each stage is, and what caused the misses (share of all captures)."""
    g = eb.groupby("condition", sort=False)
    t = pd.DataFrame({
        "cells correct (CNN)": g["share_correct"].mean(),
        "patch right (CNN)": g["patch_accuracy"].mean(),
        "offset error mm (CNN)": g["offset_error_mm"].median(),
        "chosen cells correct (RANSAC)": g["inliers_correct"].mean(),
        "located: oracle CNN (PnP limit)": g["error_oracle_cnn_mm"].apply(lambda e: (e <= tol_mm).mean()),
        "located: correct cells only": g["error_correct_cells_only_mm"].apply(lambda e: (e <= tol_mm).mean()),
        "located: full pipeline": g["error_full_mm"].apply(lambda e: (e <= tol_mm).mean()),
    })
    causes = pd.crosstab(eb["condition"], eb["cause"], normalize="index").reindex(columns=CAUSES[1:], fill_value=0.0)
    causes.columns = [f"miss: {c}" for c in causes.columns]
    return t.join(causes).reset_index()


def cell_error_image(cnn, cap: Capture, scale: int = 2, cell_mm: float = 10.0) -> np.ndarray:
    """The capture with each cell coloured by how far its predicted CAD point is from the truth (BGR):
    green within 5 mm, yellow within `cell_mm`, red beyond, grey where the CNN says 'not the part'."""
    import cv2
    c = cnn.cells(cap.image)
    true, seen = true_points(cap, c["uv"])
    err = np.linalg.norm(np.nan_to_num(cnn.points(c, 0) - true), axis=1)
    img = cv2.cvtColor(cap.image, cv2.COLOR_GRAY2BGR).astype(float) * 0.6
    step = cap.image.shape[0] / int(np.sqrt(len(err)))
    for (u, v), fg, s, e in zip(c["uv"], c["part"], seen, err):
        if not fg:
            continue
        color = (90, 90, 90) if not s else (60, 200, 60) if e <= 5 else (0, 210, 240) if e <= cell_mm else (40, 40, 220)
        x0, y0 = int(u - step / 2), int(v - step / 2)
        img[y0:y0 + int(step), x0:x0 + int(step)] = 0.45 * img[y0:y0 + int(step), x0:x0 + int(step)] + 0.55 * np.array(color)
    out = img.astype(np.uint8)
    return cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
