"""Fast checks of the geometry, the data and the evaluation (no long training)."""
from __future__ import annotations

import cv2
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


# --- predictability and interpretability: CAD fit, error budget, localisability, held-out, calibration ---

def test_cad_edge_fit_pulls_a_disturbed_pose_back(setup):
    from cadloc.fit import EdgeModel, fit_pose
    _, _, r = setup
    rng = np.random.default_rng(6)
    cap = sample_capture(r, rng, (80, 140))
    edges = EdgeModel(r)
    assert len(edges.visible(cap.R, cap.t)) > 50
    R0 = cv2.Rodrigues(cv2.Rodrigues(cap.R)[0] + np.array([[0.02], [-0.015], [0.01]]))[0]
    t0 = cap.t + np.array([2.0, -2.0, 3.0])
    f = fit_pose(edges, cap.image, R0, t0)
    before = np.linalg.norm(-R0.T @ t0 + cap.R.T @ cap.t)
    after = np.linalg.norm(-f.R.T @ f.t + cap.R.T @ cap.t)
    assert f.agreement >= f.start_agreement and after < before


def test_error_budget_gives_a_cause_per_capture(setup):
    torch = pytest.importorskip("torch")
    from cadloc.cnn import CNN, training_renders
    from cadloc.diagnose import CAUSES, budget_table, error_budget
    part, cam, r = setup
    torch.manual_seed(0)
    model = CNN(cam, part, width=8, patches=16).fit(training_renders(r, 16, 0), epochs=1, batch=8, log=lambda *_: None)
    eb = error_budget(model, r, {"nominal": make_test_set(r, "nominal", 2, seed=8)})
    assert set(eb["cause"]) <= set(CAUSES) and len(eb) == 2
    assert (eb["error_oracle_cnn_mm"] < 1.0).all()         # true points give the pose: the geometry is not the limit
    assert "located: full pipeline" in budget_table(eb).columns


def test_calibration_scores_and_one_class():
    from cadloc.calibration import Logistic, accept_rule, scores
    rng = np.random.default_rng(0)
    p = rng.random(4000)
    y = rng.random(4000) < p                               # perfectly calibrated by construction
    s = scores(p, y)
    assert s["ece"] < 0.03 and s["auc"] > 0.7
    assert accept_rule(p, y)["accepted_located"].iloc[-1] > 0.9
    lr = Logistic().fit(rng.random((10, 2)), np.zeros(10))
    assert np.allclose(lr.predict(rng.random((3, 2))), 0.0)


def test_localisability_features_are_in_range(setup):
    from cadloc.data import reference_set
    from cadloc.localisability import FEATURES, view_features
    from cadloc.methods import Retrieval
    _, _, r = setup
    ret = Retrieval(reference_set(r, 40, 2))
    cap = sample_capture(r, np.random.default_rng(7), (60, 150))
    f = view_features(r, cap.R, cap.t, ret)
    assert set(f) == set(FEATURES)
    assert 0 <= f["edges"] <= 1 and 0 < f["coverage"] <= 1 and f["orientations"] >= 0 and -1 <= f["look_alike"] <= 1


def test_out_of_range_sets(setup):
    from cadloc.validation import SETS, out_of_range
    _, _, r = setup
    r2, caps = out_of_range(r, "wide lens (65 deg)", 2, 1)
    assert r2.cam.fov_deg == 65 and len(caps) == 2
    _, far = out_of_range(r, "farther (200-300 mm)", 2, 1)
    assert all(c.meta["distance_mm"] >= 200 for c in far) and len(SETS) == 4
