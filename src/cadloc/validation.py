"""Held-out validation: does the model work outside what it was trained on?

The CNN was trained on CAD renders at 25-200 mm, with a 50 degree lens, surface texture up to 0.08 and the camera
within 20 degrees of upright. It is evaluated on
  - images it was trained on (seen),
  - new images from the same settings (unseen, same distribution),
  - the test captures (unseen, with the six test conditions), and
  - captures outside the training range: farther away, other lenses, stronger surface texture.
A small gap between seen and unseen means the model has learnt the part, not the training images; the out-of-range
rows show how far it can be pushed.
"""
from __future__ import annotations

import numpy as np

from .render import Camera, Capture, Renderer, sample_capture

# name -> (description, inside the training range?, settings)
SETS = {
    "farther (200-300 mm)": ("beyond the 200 mm training limit", False, dict(dist=(200, 300), min_visible=0.15)),
    "wide lens (65 deg)": ("training used a 50 degree lens", False, dict(dist=(60, 180), fov=65.0)),
    "narrow lens (35 deg)": ("training used a 50 degree lens", False, dict(dist=(60, 180), fov=35.0)),
    "strong texture (0.10-0.15)": ("training texture was at most 0.08", False, dict(dist=(60, 180), texture=(0.10, 0.15))),
}


def out_of_range(renderer: Renderer, name: str, n: int, seed: int, max_roll_deg: float | None = 20
                 ) -> tuple[Renderer, list[Capture]]:
    """Captures for one out-of-range set, and the renderer (camera) they were taken with."""
    s = SETS[name][2]
    r = renderer
    if "fov" in s:
        cam = renderer.cam
        r = Renderer(renderer.part, Camera(cam.width, cam.height, s["fov"]))
    rng = np.random.default_rng(seed)
    tex = s.get("texture", (0.05, 0.05))
    caps = []
    for _ in range(n):
        c = sample_capture(r, rng, s["dist"], light="camera", max_roll_deg=max_roll_deg, texture=rng.uniform(*tex),
                           min_visible=s.get("min_visible", 0.3), spot_tol_mm=3.0)   # far: a pixel is > 1 mm
        c.meta["condition"] = name
        caps.append(c)
    return r, caps
