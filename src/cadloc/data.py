"""Reference renders from the CAD model, and test captures under harder conditions."""
from __future__ import annotations

import cv2
import numpy as np

from .render import Capture, Renderer, sample_capture

# name -> how the captures differ from the reference renders
CONDITIONS = {
    "nominal":    dict(dist=(60, 180), light="camera"),
    "free_roll":  dict(dist=(60, 180), light="camera", roll=None),         # camera held at any angle
    "close_up":   dict(dist=(25, 50), light="camera"),                    # limited visual context
    "lighting":   dict(dist=(60, 180), light="any", ambient=0.12, specular=0.6),
    "occlusion":  dict(dist=(60, 180), light="camera", occlude=(0.3, 0.5)),  # partial visibility
    "noise_blur": dict(dist=(60, 180), light="camera", noise=0.06, blur=1.5, gain=(0.7, 1.3)),
}


def reference_set(renderer: Renderer, n: int, seed: int, dist=(40, 200), max_roll_deg: float | None = 20) -> list[Capture]:
    """Views rendered from the CAD model with known pose: the only 'training data' the methods get. Plain CAD, so no
    surface texture."""
    rng = np.random.default_rng(seed)
    return [sample_capture(renderer, rng, dist, light="camera" if i % 2 else "any", max_roll_deg=max_roll_deg)
            for i in range(n)]


def _occlude(img: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Cover about `frac` of the image with clutter (a hand, a fixture), keeping the centre visible."""
    h, w = img.shape
    out = img.copy()
    covered = np.zeros_like(img, dtype=bool)
    while covered.mean() < frac:
        side = rng.integers(4)
        bw, bh = int(rng.uniform(0.3, 0.6) * w), int(rng.uniform(0.3, 0.6) * h)
        x0 = [0, w - bw, rng.integers(0, w - bw), rng.integers(0, w - bw)][side]
        y0 = [rng.integers(0, h - bh), rng.integers(0, h - bh), 0, h - bh][side]
        box = np.zeros_like(covered)
        box[y0:y0 + bh, x0:x0 + bw] = True
        box[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8] = False
        ramp = np.linspace(0, 1, bw)[None, :] * rng.uniform(-0.3, 0.3) + rng.uniform(0.15, 0.8)
        clutter = (np.clip(ramp + rng.normal(0, 0.08, (bh, bw)), 0, 1) * 255).astype(np.uint8)
        out[y0:y0 + bh, x0:x0 + bw] = np.where(box[y0:y0 + bh, x0:x0 + bw], clutter, out[y0:y0 + bh, x0:x0 + bw])
        covered |= box
    return out


def test_set(renderer: Renderer, condition: str, n: int, seed: int, max_roll_deg: float | None = 20,
             texture: float = 0.05) -> list[Capture]:
    """'Real' captures: random surface texture the CAD does not have, plus the condition's difficulty."""
    c = CONDITIONS[condition]
    rng = np.random.default_rng(seed)
    kw = {k: c[k] for k in ("ambient", "specular", "noise") if k in c}
    roll = c.get("roll", max_roll_deg)
    caps = []
    for _ in range(n):
        cap = sample_capture(renderer, rng, c["dist"], light=c["light"], max_roll_deg=roll, texture=texture, **kw)
        img = cap.image
        if "occlude" in c:
            img = _occlude(img, rng.uniform(*c["occlude"]), rng)
        if "blur" in c:
            img = cv2.GaussianBlur(img, (0, 0), c["blur"])
        if "gain" in c:
            img = np.clip(img.astype(float) * rng.uniform(*c["gain"]), 0, 255).astype(np.uint8)
        cap.image = img
        cap.meta["condition"] = condition
        caps.append(cap)
    return caps
