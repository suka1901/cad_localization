"""Score each method on each test condition against the true pose and surface point."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .render import Capture, Renderer


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(R1 @ R2.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def centre_point(renderer: Renderer, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Where the image centre lands on the CAD surface for a pose: the data point to attach."""
    cam = renderer.cam
    pts, prim, hit, _ = renderer.raycast(R, t)
    i = (cam.height // 2) * cam.width + cam.width // 2
    if not hit[i]:
        return None, -1
    return pts[i], int(renderer.part.face_region[prim[i]])


def run(methods: list, conditions: dict[str, list[Capture]], renderer: Renderer) -> pd.DataFrame:
    rows = []
    for cond, caps in conditions.items():
        for i, cap in enumerate(caps):
            for m in methods:
                t0 = time.perf_counter()
                est = m.locate(cap.image)
                ms = 1000 * (time.perf_counter() - t0)
                row = {"condition": cond, "capture": i, "method": m.name, "ms": ms, "confidence": est.inliers,
                       "true_region": cap.target_region, "distance_mm": cap.meta["distance_mm"],
                       "found": False, "centre_error_mm": np.nan, "rotation_error_deg": np.nan,
                       "camera_error_mm": np.nan, "region_correct": False, "est_x": np.nan, "est_y": np.nan,
                       "est_z": np.nan, "true_x": cap.target[0], "true_y": cap.target[1], "true_z": cap.target[2]}
                if est.R is not None and est.t is not None:
                    p, reg = centre_point(renderer, est.R, est.t)
                    row["rotation_error_deg"] = rotation_error_deg(est.R, cap.R)
                    row["camera_error_mm"] = float(np.linalg.norm(-est.R.T @ est.t + cap.R.T @ cap.t))
                    if p is not None:
                        row.update(found=True, centre_error_mm=float(np.linalg.norm(p - cap.target)),
                                   region_correct=reg == cap.target_region, est_x=p[0], est_y=p[1], est_z=p[2])
                rows.append(row)
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, tol_mm: float = 10.0, strict_mm: float = 5.0) -> pd.DataFrame:
    d = df.assign(within=lambda x: x["centre_error_mm"] <= tol_mm, strict=lambda x: x["centre_error_mm"] <= strict_mm)
    g = d.groupby(["condition", "method"])
    out = pd.DataFrame({
        f"located_within_{tol_mm:g}mm": g["within"].mean(),
        f"located_within_{strict_mm:g}mm": g["strict"].mean(),
        "region_correct": g["region_correct"].mean(),
        "median_error_mm": g["centre_error_mm"].median(),
        "median_rotation_deg": g["rotation_error_deg"].median(),
        "ms_per_capture": g["ms"].median(),
    }).reset_index()
    return out
