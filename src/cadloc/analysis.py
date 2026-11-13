"""Make the results predictable and interpretable: the four analyses that follow the main evaluation.

  error budget      which stage caused each miss (diagnose.py)
  calibration       a probability of being right that holds on new captures (calibration.py)
  localisability    where on the part captures will be hard, from the CAD model alone (localisability.py)
  held-out          seen vs unseen vs out-of-range captures (validation.py)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from . import calibration as cal
from . import localisability as loc
from .data import CONDITIONS, test_set
from .diagnose import budget_table, error_budget
from .evaluate import run as evaluate
from .methods import Retrieval
from .render import Capture, Renderer
from .validation import SETS, out_of_range

GEOMETRY_CONDITIONS = ["nominal", "lighting", "noise_blur"]   # same viewpoints; only the image quality differs
# is each test condition inside what the CNN saw in training (25-200 mm, roll within 20 deg, any light, clutter,
# noise and blur)?
IN_TRAINING = {"nominal": (True, "test captures"), "lighting": (True, "test captures, other lighting"),
               "noise_blur": (True, "test captures, noise and blur"),
               "close_up": (True, "25-50 mm: in range, but little of the part in view"),
               "occlusion": (True, "30-50 % covered: more than most training clutter"),
               "free_roll": (False, "any camera roll; training was within 20 deg of upright")}


def _sel(df: pd.DataFrame, col: str, value) -> pd.DataFrame:
    return cast(pd.DataFrame, df[df[col] == value])


def cached(path: Path, compute, resume: bool) -> pd.DataFrame:
    """Reuse a finished stage's table when resuming (after a restart or a lost Colab session)."""
    if resume and path.exists():
        print(f"  reusing {path.name}")
        return pd.read_csv(path)
    d = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(path, index=False)
    return d


def _located(df: pd.DataFrame, tol: float) -> np.ndarray:
    return (df["centre_error_mm"] <= tol).to_numpy()


def _rate(df: pd.DataFrame, method: str, tol: float) -> tuple[float, float]:
    e = _sel(df, "method", method)["centre_error_mm"].to_numpy(dtype=float)
    return float(np.mean(e <= tol)), float(np.nanmedian(e)) if np.isfinite(e).any() else float("nan")


def analyse(cfg: dict, renderer: Renderer, cnn, fitted, retrieval: Retrieval, tests: dict[str, list[Capture]],
            df: pd.DataFrame, out: Path, log=print, resume: bool = False) -> dict:
    a, t = cfg["analysis"], cfg["test"]
    tol, roll = t["tolerance_mm"], cfg["references"]["max_roll_deg"]
    res: dict = {}

    # 1. error budget of the CNN pipeline on the test captures
    log("analysis: error budget")
    eb = cached(out / "error_budget.csv", lambda: error_budget(cnn, renderer, tests, tol), resume)
    res["budget"] = budget_table(eb, tol)
    res["budget"].to_csv(out / "error_budget_summary.csv", index=False)

    # 2. calibration captures: new seeds, all six conditions, never used for testing
    log("analysis: calibration captures")
    n_cal = a["calibration_per_condition"]
    calib = {name: test_set(renderer, name, n_cal, a["calibration_seed"] + i, max_roll_deg=roll, texture=t["texture"])
             for i, name in enumerate(CONDITIONS)}
    dc = cached(out / "calibration_captures.csv", lambda: evaluate([fitted], calib, renderer), resume)
    dt = _sel(df, "method", fitted.name).copy()
    yc, yt = _located(dc, tol), _located(dt, tol)
    full = cal.Calibrator().fit(dc, yc)
    inl = cal.Calibrator(("inliers",)).fit(dc, yc)
    p_full, p_inl = full.predict(dt), inl.predict(dt)
    df.loc[dt.index, "p_located"] = p_full
    res["calibration"] = {
        "with_cad_fit": cal.scores(p_full, yt), "inliers_only": cal.scores(p_inl, yt),
        "reliability": cal.reliability(p_full, yt), "reliability_inliers": cal.reliability(p_inl, yt),
        "accept": cal.accept_rule(p_full, yt), "effects": full.effects(),
        "calibration_captures": len(dc)}
    res["calibration"]["reliability"].to_csv(out / "calibration_reliability.csv", index=False)

    # 3. localisability from the CAD model alone, fitted on calibration captures, tested on test captures
    log("analysis: localisability (CAD only)")
    caps_c = [c for n in GEOMETRY_CONDITIONS for c in calib[n]]
    caps_t = [c for n in GEOMETRY_CONDITIONS for c in tests[n]]
    yc_g = np.concatenate([_located(_sel(dc, "condition", n), tol) for n in GEOMETRY_CONDITIONS])
    yt_g = np.concatenate([_located(_sel(dt, "condition", n), tol) for n in GEOMETRY_CONDITIONS])
    Xc, Xt = loc.features_for(renderer, caps_c, retrieval), loc.features_for(renderer, caps_t, retrieval)
    lm = loc.Localisability().fit(Xc, yc_g)
    pt = lm.predict(Xt)
    spots = loc.surface_map(renderer, lm, retrieval, a["map_spots"], a["map_seed"], max_roll_deg=roll)
    spots.to_csv(out / "localisability_map.csv", index=False)
    tg = pd.DataFrame({"region": [c.target_region for c in caps_t], "observed": yt_g, "predicted": pt})
    regions = cast(pd.DataFrame, tg.groupby("region").agg(captures=("observed", "size"), observed=("observed", "mean"),
                                                          predicted_from_test_views=("predicted", "mean")))
    regions["predicted_from_cad_map"] = spots.groupby("region")["p_located"].mean()
    regions = regions.reset_index()
    ok = regions[["observed", "predicted_from_cad_map"]].dropna().to_numpy(dtype=float)
    regions.to_csv(out / "localisability_regions.csv", index=False)
    res["localisability"] = {
        "auc": cal.auc(yt_g, pt), "single_auc": loc.single_feature_auc(Xt, yt_g),
        "effects": lm.effects(), "regions": regions, "spots": spots,
        "face_p": loc.face_values(renderer, spots),
        "region_corr": float(np.corrcoef(ok.T)[0, 1]) if len(ok) > 2 else float("nan")}

    # 4. held-out: seen, unseen same settings, test, out of range
    log("analysis: held-out validation")
    from .cnn import training_renders
    n_h, c = a["heldout_per_set"], cfg["cnn"]
    rows = []

    def add(name: str, inside: bool, note: str, d: pd.DataFrame):
        for m in (cnn.name, fitted.name):
            rate, med = _rate(d, m, tol)
            rows.append({"set": name, "inside_training_range": inside, "note": note, "method": m,
                         "captures": int((d["method"] == m).sum()), "located": rate, "median_error_mm": med})

    runs = out / "heldout_runs"
    add("training images (seen)", True, "rendered exactly as in training",
        cached(runs / "seen.csv", lambda: evaluate(
            [cnn, fitted], {"seen": training_renders(renderer, n_h, c["seed"], max_roll_deg=roll)}, renderer), resume))
    add("new images, training settings (unseen)", True, "same settings, new random draws",
        cached(runs / "fresh.csv", lambda: evaluate(
            [cnn, fitted], {"fresh": training_renders(renderer, n_h, c["seed"] + 1000, max_roll_deg=roll)}, renderer),
            resume))
    for name in CONDITIONS:
        add(f"test: {name}", IN_TRAINING[name][0], IN_TRAINING[name][1], _sel(df, "condition", name))
    for i, name in enumerate(SETS):
        def compute(i=i, name=name) -> pd.DataFrame:
            r2, caps = out_of_range(renderer, name, n_h, a["heldout_seed"] + i, max_roll_deg=roll)
            f2 = type(fitted)(cnn, r2, fitted.anchor_weight, fitted.stages, fitted.hypotheses)
            cam = cnn.cam
            cnn.cam = r2.cam                                 # PnP with the lens the capture was taken with
            try:
                return evaluate([cnn, f2], {name: caps}, r2)
            finally:
                cnn.cam = cam
        add(name, False, SETS[name][0], cached(runs / f"out_of_range_{i}.csv", compute, resume))
    res["heldout"] = pd.DataFrame(rows)
    res["heldout"].to_csv(out / "heldout.csv", index=False)

    summary = {
        "calibration": {k: v for k, v in res["calibration"].items() if isinstance(v, (dict, int))},
        "localisability": {"auc": res["localisability"]["auc"], "single_auc": res["localisability"]["single_auc"],
                           "region_corr": res["localisability"]["region_corr"]},
    }
    (out / "analysis.json").write_text(json.dumps(summary, indent=1), encoding="utf8")
    return res
