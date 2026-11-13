"""Confidence that can be trusted: the probability that a result is within the tolerance.

The raw signals (RANSAC inliers from the CNN, and how well the CAD edges agree with the image after the fit) are
turned into a probability by a logistic model fitted on separate calibration captures (new seeds, never used for
testing). It is then checked on the test captures: when the model says 90 %, are 90 % of those results right?

  reliability   predicted probability against the observed share located, in bins
  ECE           expected calibration error: the average gap between the two, weighted by bin size (0 is perfect)
  Brier score   mean squared error of the probability (lower is better)
  AUC           how well the probability ranks right above wrong results (0.5 = chance, 1 = perfect)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

SIGNALS = {"inliers": "RANSAC inliers (log)", "agreement": "CAD edges found in the image (share)",
           "residual": "CAD edge distance (px, median)"}


def signals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"inliers": np.log1p(df["confidence"].to_numpy(dtype=float)),
                         "agreement": df["edge_agreement"].fillna(0.0).to_numpy(dtype=float),
                         "residual": np.minimum(df["fit_residual_px"].fillna(20.0).to_numpy(dtype=float), 20.0)})


class Logistic:
    """Standardised logistic regression that also copes with a single class (then a constant probability)."""

    def __init__(self):
        self.scaler, self.model, self.const = StandardScaler(), LogisticRegression(C=1.0), None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Logistic":
        Xs = self.scaler.fit_transform(X)
        if len(np.unique(y)) < 2:
            self.const = float(np.mean(y))
        else:
            self.model.fit(Xs, y.astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.const is not None:
            return np.full(len(X), self.const)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def coef(self) -> np.ndarray:
        return np.zeros(len(self.scaler.scale_)) if self.const is not None else self.model.coef_[0]  # pyright: ignore[reportArgumentType]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


class Calibrator:
    def __init__(self, use: tuple[str, ...] = ("inliers", "agreement", "residual")):
        self.use = list(use)
        self.lr = Logistic()

    def fit(self, df: pd.DataFrame, located: np.ndarray) -> "Calibrator":
        self.lr.fit(signals(df)[self.use].to_numpy(), located)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict(signals(df)[self.use].to_numpy())

    def effects(self) -> pd.DataFrame:
        return pd.DataFrame({"signal": [SIGNALS[u] for u in self.use], "effect_per_sd": self.lr.coef()})


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = [{"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "predicted": float(p[b == i].mean()),
             "observed": float(y[b == i].mean()), "captures": int((b == i).sum())}
            for i in range(bins) if (b == i).any()]
    return pd.DataFrame(rows)


def scores(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    rel = reliability(p, y)
    ece = float((rel["captures"] * (rel["predicted"] - rel["observed"]).abs()).sum() / len(p))
    return {"ece": ece, "brier": float(brier_score_loss(y, p, labels=[0, 1])), "auc": auc(y, p)}


def accept_rule(p: np.ndarray, y: np.ndarray, thresholds=(0.5, 0.8, 0.9)) -> pd.DataFrame:
    """Store a result automatically when its probability is at least the threshold: how many, and how many right."""
    return pd.DataFrame([{"accept_if_p_at_least": th, "share_accepted": float((p >= th).mean()),
                          "accepted_located": float(y[p >= th].mean()) if (p >= th).any() else np.nan}
                         for th in thresholds])
