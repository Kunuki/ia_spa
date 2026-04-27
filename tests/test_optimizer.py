"""
tests/test_optimizer.py
-----------------------
Unit and integration tests for IA-SPA.

Unit tests (no GPU, no Sionna required)
----------------------------------------
- w_bar                 -- mathematical properties
- get_candidate_locations (mode="user")  -- shape validation, passthrough
- get_candidate_locations (mode="grid")  -- correct grid, uses scene bbox
- TowerOptimizer        -- loading, aggregation, S(T) properties
- run_greedy            -- output files, counts, determinism, warm-start

Integration tests (require Sionna + GPU)
-----------------------------------------
Marked with @integration; skipped unless SIONNA_INTEGRATION=1 is set.
These load the actual built-in SF and Florence scenes and verify that
get_candidate_locations returns sensible results.

Usage
-----
    # Unit tests only (fast, no GPU needed):
    pytest tests/

    # Including integration tests (GPU required):
    SIONNA_INTEGRATION=1 pytest tests/
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ia_spa.basis_functions import (
    _candidates_grid,
    _candidates_user,
    get_candidate_locations,
)
from ia_spa.optimizer import TowerOptimizer, run_greedy, w_bar


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

GRID_H, GRID_W = 20, 20
N_CANDS = 6


def _make_fake_scene(bbox_min=(0.0, 0.0, 0.0), bbox_max=(100.0, 80.0, 50.0)):
    """Return a minimal fake scene object whose _scene.bbox() works."""
    lo = np.asarray(bbox_min)
    hi = np.asarray(bbox_max)
    bbox = SimpleNamespace(min=lo, max=hi)
    scene = SimpleNamespace(_scene=SimpleNamespace(bbox=lambda: bbox))
    return scene, lo, hi


@pytest.fixture()
def fake_basis_dir(tmp_path):
    """Minimal basis-function folder with synthetic rate maps."""
    sionna_dir = tmp_path / "Sionna"
    sionna_dir.mkdir()

    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(N_CANDS, 3))

    with open(sionna_dir / "0_Coordinates.txt", "w") as fh:
        for i, (x, y, z) in enumerate(coords):
            fh.write(f"{i}, {x}, {y}, {z}\n")

    for i in range(N_CANDS):
        rate = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        r0 = rng.integers(0, GRID_H - 5)
        c0 = rng.integers(0, GRID_W - 5)
        rate[r0: r0 + 5, c0: c0 + 5] = rng.uniform(1e6, 1e8)
        np.save(sionna_dir / f"{i}.npy", rate)

    return tmp_path


# ===========================================================================
# 1. w_bar
# ===========================================================================

class TestWBar:
    def test_non_negative(self):
        x = np.linspace(0, 1e9, 500)
        assert np.all(w_bar(x) >= 0)

    def test_strictly_below_one(self):
        assert np.all(w_bar(np.linspace(0, 1e12, 500)) < 1)

    def test_zero_at_origin(self):
        assert w_bar(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_monotone_increasing(self):
        x = np.linspace(0, 1e9, 500)
        assert np.all(np.diff(w_bar(x)) >= 0)

    def test_concave(self):
        x = np.linspace(1, 1e9, 500)
        assert np.all(np.diff(w_bar(x), n=2) <= 1e-15)

    def test_custom_c(self):
        # w_bar(c, c) = c/(c+c) = 0.5
        assert w_bar(np.array([3.0]), c=3.0)[0] == pytest.approx(0.5)


# ===========================================================================
# 2. get_candidate_locations -- mode="user"
# ===========================================================================

class TestCandidatesUser:
    def test_returned_unchanged(self):
        scene, *_ = _make_fake_scene()
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = get_candidate_locations(scene, mode="user", user_positions=pts)
        np.testing.assert_array_equal(result, pts)

    def test_preserves_dtype_as_float(self):
        scene, *_ = _make_fake_scene()
        pts = np.array([[0, 0, 0], [1, 1, 1]])   # int input
        result = get_candidate_locations(scene, mode="user", user_positions=pts)
        assert result.dtype == float

    def test_none_raises(self):
        scene, *_ = _make_fake_scene()
        with pytest.raises(ValueError, match="user_positions"):
            get_candidate_locations(scene, mode="user")

    def test_wrong_shape_raises(self):
        scene, *_ = _make_fake_scene()
        with pytest.raises(ValueError, match="shape"):
            get_candidate_locations(scene, mode="user",
                                    user_positions=np.ones((5, 2)))

    def test_single_point_accepted(self):
        scene, *_ = _make_fake_scene()
        pt = np.array([[10.0, 20.0, 5.0]])
        result = get_candidate_locations(scene, mode="user", user_positions=pt)
        assert result.shape == (1, 3)

    def test_no_filtering_applied(self):
        """All supplied points are returned, even if they might be inside geometry."""
        scene, *_ = _make_fake_scene()
        # Deliberately put points at bbox centre and outside -- both should come back
        pts = np.array([
            [50.0, 40.0, 25.0],   # centre of domain
            [-999.0, -999.0, -999.0],  # far outside
        ])
        result = get_candidate_locations(scene, mode="user", user_positions=pts)
        assert len(result) == 2


# ===========================================================================
# 3. get_candidate_locations -- mode="grid"
# ===========================================================================

class TestCandidatesGrid:
    def test_output_shape(self):
        scene, lo, hi = _make_fake_scene()
        result = get_candidate_locations(scene, mode="grid", dx=25, dy=25, dz=25)
        assert result.ndim == 2
        assert result.shape[1] == 3

    def test_covers_bbox_min_to_max(self):
        scene, lo, hi = _make_fake_scene()
        result = get_candidate_locations(scene, mode="grid", dx=20, dy=20, dz=20)
        # X range
        assert result[:, 0].min() == pytest.approx(lo[0])
        assert result[:, 0].max() == pytest.approx(hi[0], abs=20)
        # Y range
        assert result[:, 1].min() == pytest.approx(lo[1])

    def test_z_range_within_bbox(self):
        scene, lo, hi = _make_fake_scene()
        result = get_candidate_locations(scene, mode="grid", dx=20, dy=20, dz=15)
        assert result[:, 2].min() >= lo[2] - 1e-9
        assert result[:, 2].max() <= hi[2] + 15 + 1e-9  # last step may reach hi

    def test_correct_node_count(self):
        """Grid count should match ceil((range / step) + 1) in each axis."""
        lo = (0.0, 0.0, 0.0)
        hi = (100.0, 80.0, 60.0)
        scene, _, _ = _make_fake_scene(lo, hi)
        dx, dy, dz = 25.0, 20.0, 30.0
        result = get_candidate_locations(scene, mode="grid", dx=dx, dy=dy, dz=dz)
        nx = len(np.arange(0, 100 + dx * 0.5, dx))  # 0,25,50,75,100 -> 5
        ny = len(np.arange(0, 80 + dy * 0.5, dy))   # 0,20,40,60,80  -> 5
        nz = len(np.arange(0, 60 + dz * 0.5, dz))   # 0,30,60        -> 3
        assert len(result) == nx * ny * nz

    def test_step_affects_count(self):
        """Halving the step size should roughly 8x the number of candidates."""
        scene, *_ = _make_fake_scene()
        n_coarse = len(get_candidate_locations(scene, mode="grid", dx=20, dy=20, dz=20))
        n_fine   = len(get_candidate_locations(scene, mode="grid", dx=10, dy=10, dz=10))
        assert n_fine > n_coarse

    def test_no_filtering(self):
        """All grid points are returned regardless of obstacle position."""
        scene, lo, hi = _make_fake_scene()
        n = len(get_candidate_locations(scene, mode="grid", dx=25, dy=25, dz=25))
        # Count what we expect manually
        xs = np.arange(lo[0], hi[0] + 25 * 0.5, 25)
        ys = np.arange(lo[1], hi[1] + 25 * 0.5, 25)
        zs = np.arange(lo[2], hi[2] + 25 * 0.5, 25)
        assert n == len(xs) * len(ys) * len(zs)

    def test_unknown_mode_raises(self):
        scene, *_ = _make_fake_scene()
        with pytest.raises(ValueError, match="Unknown mode"):
            get_candidate_locations(scene, mode="voxel")

    def test_grid_helper_directly(self):
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([40.0, 40.0, 20.0])
        result = _candidates_grid(lo, hi, dx=20, dy=20, dz=10)
        # x: 0,20,40 -> 3; y: 0,20,40 -> 3; z: 0,10,20 -> 3
        assert len(result) == 3 * 3 * 3

    def test_user_helper_directly(self):
        pts = np.array([[1.0, 2.0, 3.0]])
        result = _candidates_user(pts)
        np.testing.assert_array_equal(result, pts)


# ===========================================================================
# 4. TowerOptimizer
# ===========================================================================

class TestTowerOptimizer:
    def test_loads_coordinates(self, fake_basis_dir):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        assert opt.n_candidates == N_CANDS
        assert opt.coords.shape == (N_CANDS, 3)

    def test_invalid_aggregation_raises(self, fake_basis_dir):
        with pytest.raises(ValueError, match="aggregation"):
            TowerOptimizer(fake_basis_dir, aggregation="mean")

    def test_missing_folder_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TowerOptimizer(tmp_path / "nonexistent")

    def test_s_empty_is_zero(self, fake_basis_dir):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        assert opt.network_quality(np.zeros(N_CANDS)) == pytest.approx(0.0)

    def test_s_monotone(self, fake_basis_dir):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        u1 = np.zeros(N_CANDS); u1[0] = 1
        u2 = u1.copy(); u2[1] = 1
        assert opt.network_quality(u1) <= opt.network_quality(u2)

    def test_s_bounded(self, fake_basis_dir):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        assert 0 <= opt.network_quality(np.ones(N_CANDS)) < 1

    def test_marginal_gains_shape_and_sign(self, fake_basis_dir):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        gains = opt.marginal_gains(np.zeros(N_CANDS))
        assert gains.shape == (N_CANDS,)
        assert np.all(gains >= 0)

    def test_marginal_gains_sum_mode(self, fake_basis_dir):
        opt = TowerOptimizer(fake_basis_dir, aggregation="sum")
        assert np.all(opt.marginal_gains(np.zeros(N_CANDS)) >= 0)

    def test_idempotent_max_gain(self, fake_basis_dir):
        """Re-adding a selected candidate gives zero marginal gain under P_MAX."""
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        u = np.zeros(N_CANDS); u[0] = 1
        assert opt.marginal_gains(u)[0] == pytest.approx(0.0, abs=1e-10)

    def test_diminishing_returns(self, fake_basis_dir):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        u = np.zeros(N_CANDS)
        best = []
        for _ in range(4):
            g = opt.marginal_gains(u)
            best.append(g.max())
            u[np.argmax(g)] += 1
        for i in range(len(best) - 1):
            assert best[i] >= best[i + 1] - 1e-12


# ===========================================================================
# 5. run_greedy
# ===========================================================================

class TestRunGreedy:
    def test_output_files_created(self, fake_basis_dir, tmp_path):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        run_greedy(opt, results_folder=tmp_path / "r", n_iter=3)
        r = tmp_path / "r"
        assert (r / "Locations.txt").exists()
        assert (r / "u.npy").exists()
        assert (r / "Gain_Function_0.npy").exists()
        assert (r / "Gain_Function_2.npy").exists()

    def test_locations_count(self, fake_basis_dir, tmp_path):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        run_greedy(opt, results_folder=tmp_path / "r", n_iter=4)
        lines = (tmp_path / "r" / "Locations.txt").read_text().strip().split("\n")
        assert len(lines) == 4

    def test_locations_are_xyz(self, fake_basis_dir, tmp_path):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        run_greedy(opt, results_folder=tmp_path / "r", n_iter=2)
        for line in (tmp_path / "r" / "Locations.txt").read_text().strip().split("\n"):
            assert len([float(v) for v in line.split(", ")]) == 3

    def test_u_vector_sum(self, fake_basis_dir, tmp_path):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        run_greedy(opt, results_folder=tmp_path / "r", n_iter=3)
        u = np.load(tmp_path / "r" / "u.npy")
        assert u.shape == (N_CANDS,)
        assert int(u.sum()) == 3

    def test_network_quality_improves(self, fake_basis_dir, tmp_path):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        run_greedy(opt, results_folder=tmp_path / "r", n_iter=4)
        total_gain = sum(
            np.load(tmp_path / "r" / f"Gain_Function_{k}.npy").max()
            for k in range(4)
        )
        assert total_gain > 0

    def test_warm_start_u_sum(self, fake_basis_dir, tmp_path):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        run_greedy(opt, tmp_path / "r", n_iter=2, fixed_towers=opt.coords[:1])
        assert int(np.load(tmp_path / "r" / "u.npy").sum()) == 3  # 1 fixed + 2

    def test_reproducible(self, fake_basis_dir, tmp_path):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        run_greedy(opt, tmp_path / "r1", n_iter=3)
        run_greedy(opt, tmp_path / "r2", n_iter=3)
        assert (tmp_path / "r1" / "Locations.txt").read_text() == \
               (tmp_path / "r2" / "Locations.txt").read_text()

    def test_overwrite_on_rerun(self, fake_basis_dir, tmp_path):
        opt = TowerOptimizer(fake_basis_dir, aggregation="max")
        run_greedy(opt, tmp_path / "r", n_iter=2)
        run_greedy(opt, tmp_path / "r", n_iter=3)
        lines = (tmp_path / "r" / "Locations.txt").read_text().strip().split("\n")
        assert len(lines) == 3


# ===========================================================================
# Integration tests -- require a GPU and Sionna installation
# ===========================================================================

integration = pytest.mark.skipif(
    os.environ.get("SIONNA_INTEGRATION") != "1",
    reason="Set SIONNA_INTEGRATION=1 to run GPU/Sionna integration tests.",
)


@integration
def test_sf_grid_candidates():
    """San Francisco: grid mode returns the right shape and covers the bbox."""
    from sionna.rt import load_scene, scene as sionna_scenes

    sf = load_scene(sionna_scenes.san_francisco, merge_shapes=False)
    candidates = get_candidate_locations(sf, mode="grid", dx=30, dy=30, dz=20)

    assert candidates.ndim == 2
    assert candidates.shape[1] == 3
    assert len(candidates) > 100

    bbox = sf._scene.bbox()
    assert candidates[:, 0].min() >= bbox.min[0] - 1e-6
    assert candidates[:, 1].min() >= bbox.min[1] - 1e-6


@integration
def test_fl_grid_candidates():
    """Florence: grid mode works without any terrain knowledge."""
    from sionna.rt import load_scene, scene as sionna_scenes

    fl = load_scene(sionna_scenes.florence, merge_shapes=False)
    candidates = get_candidate_locations(fl, mode="grid", dx=20, dy=20, dz=15)

    assert candidates.ndim == 2
    assert candidates.shape[1] == 3
    assert len(candidates) > 50


@integration
def test_sf_user_candidates():
    """San Francisco: user-supplied positions are returned unchanged."""
    from sionna.rt import load_scene, scene as sionna_scenes

    sf = load_scene(sionna_scenes.san_francisco, merge_shapes=False)
    bbox = sf._scene.bbox()
    centre = (np.array(bbox.min) + np.array(bbox.max)) / 2

    my_positions = np.array([
        centre,
        centre + np.array([10, 0, 20]),
        centre + np.array([-10, 5, 10]),
    ])
    candidates = get_candidate_locations(sf, mode="user", user_positions=my_positions)

    assert candidates.shape == (3, 3)
    np.testing.assert_array_equal(candidates, my_positions)


@integration
def test_fl_user_single_point():
    """Florence: a single user-supplied point is accepted."""
    from sionna.rt import load_scene, scene as sionna_scenes

    fl = load_scene(sionna_scenes.florence, merge_shapes=False)
    bbox = fl._scene.bbox()
    pt = np.array([[(bbox.min[0] + bbox.max[0]) / 2,
                    (bbox.min[1] + bbox.max[1]) / 2,
                    bbox.max[2] + 10]])  # above the scene
    candidates = get_candidate_locations(fl, mode="user", user_positions=pt)
    assert candidates.shape == (1, 3)
