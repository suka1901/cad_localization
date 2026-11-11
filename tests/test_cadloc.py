"""Fast checks of the geometry, the data and the evaluation (no long training)."""
from __future__ import annotations

import numpy as np
import pytest

from cadloc.data import CONDITIONS
from cadloc.data import test_set as make_test_set
from cadloc.evaluate import centre_point, rotation_error_deg, run, summary
from cadloc.methods import Estimate, _pnp, foreground, pixel_features
from cadloc.part import load_part
from cadloc.render import UP, Camera, Renderer, look_at, sample_capture


@pytest.fixture(scope="module")
def setup():
    part = load_part(None, 8)
    cam = Camera(96, 96, 50)
    return part, cam, Renderer(part, cam)


def test_part_is_a_closed_solid_with_regions(setup):
    part, _, _ = setup
    assert part.mesh.is_watertight
    assert part.n_regions == 8
    assert set(np.unique(part.face_region)) <= set(range(8))
    assert np.allclose(part.mesh.bounds.mean(axis=0), 0, atol=1e-6)


def test_rendered_points_project_back_to_their_pixels(setup):
    _, cam, r = setup
    cap = sample_capture(r, np.random.default_rng(0), (60, 150))
    ys, xs = np.nonzero(cap.mask)
    pts = cap.xyz[ys, xs]
    proj = cam.K @ (cap.R @ pts.T + cap.t[:, None])
    uv = (proj[:2] / proj[2]).T
    assert np.abs(uv - np.stack([xs, ys], 1) - 0.5).max() < 1e-3


def test_pnp_recovers_pose_from_true_correspondences(setup):
    _, cam, r = setup
    rng = np.random.default_rng(1)
    cap = sample_capture(r, rng, (60, 150))
    ys, xs = np.nonzero(cap.mask)
    pick = rng.choice(len(xs), 100, replace=False)
    uv = np.stack([xs[pick], ys[pick]], 1) + 0.5
    est = _pnp(uv, cap.xyz[ys[pick], xs[pick]], cam.K)
    assert est.R is not None and est.t is not None
    assert rotation_error_deg(est.R, cap.R) < 0.1
    p, region = centre_point(r, est.R, est.t)
    assert p is not None and np.linalg.norm(p - cap.target) < 0.5 and region == cap.target_region


def test_roll_prior_keeps_the_camera_upright():
    rng = np.random.default_rng(2)
    R, _ = look_at(np.array([0.0, -200, 30]), np.zeros(3), rng, max_roll_deg=0)
    assert R[1] @ UP < -0.95                 # image y axis (down) points against part "up"


def test_conditions_produce_valid_captures(setup):
    _, cam, r = setup
    for name in CONDITIONS:
        caps = make_test_set(r, name, 2, seed=5)
        for c in caps:
            assert c.image.shape == (cam.height, cam.width) and c.target_region >= 0
            assert c.meta["condition"] == name


def test_oracle_method_scores_perfectly(setup):
    _, _, r = setup
    caps = make_test_set(r, "nominal", 3, seed=9)
    lookup = {c.image.tobytes(): c for c in caps}

    class Oracle:
        name = "oracle"

        def locate(self, img):
            c = lookup[img.tobytes()]
            return Estimate(c.R, c.t, 99)

    s = summary(run([Oracle()], {"nominal": caps}, r), tol_mm=1.0, strict_mm=0.5)
    assert s["located_within_1mm"].iloc[0] == 1.0 and s["located_within_0.5mm"].iloc[0] == 1.0
    assert s["region_correct"].iloc[0] == 1.0


def test_pixel_features_and_foreground(setup):
    _, _, r = setup
    cap = sample_capture(r, np.random.default_rng(3), (60, 150))
    uv = np.array([[10.5, 10.5], [48.5, 48.5]])
    a = pixel_features(cap.image, uv, oriented=False)
    b = pixel_features(cap.image, uv, oriented=True)
    assert a.shape[0] == 2 and b.shape[1] > a.shape[1] and np.isfinite(b).all()
    assert (foreground(cap.image) == cap.mask).mean() > 0.9


def test_cnn_trains_and_predicts(setup):
    torch = pytest.importorskip("torch")
    from cadloc.cnn import CNN, training_renders
    part, cam, r = setup
    torch.manual_seed(0)
    model = CNN(cam, part, width=8, patches=16).fit(training_renders(r, 16, 0), epochs=1, batch=8, log=lambda *_: None)
    uv, xyz = model.correspondences(sample_capture(r, np.random.default_rng(4), (60, 150)).image)
    assert uv.shape[0] == xyz.shape[0] and xyz.shape[1] == 3
    assert len(model.history) == 1


def test_captures_are_repeatable(setup):
    _, _, r = setup
    a = make_test_set(r, "occlusion", 3, seed=11)
    b = make_test_set(r, "occlusion", 3, seed=11)
    assert all(np.array_equal(x.image, y.image) and np.allclose(x.target, y.target) for x, y in zip(a, b))
