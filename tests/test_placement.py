"""
tests/test_placement.py
-----------------------
Tests for the file-in / file-out placement path: radio maps + candidate
locations in, tower locations out.  No Sionna and no GPU required.

Coverage
--------
- Loading radio maps from .npy / .npz / text / directory layouts.
- Loading candidate locations from .npy / .csv / .txt, including the
  ``i, x, y, z`` manifest format and (x, y)-only tables.
- Input validation: count mismatches, ragged maps, NaNs, dB-scale mistakes.
- Accuracy of the greedy loop, checked three independent ways:
    * against a naive, obviously-correct reference implementation;
    * against the exhaustive optimum on small instances, including the
      (1 - 1/e) submodular guarantee of Theorem III.2;
    * against hand-built instances whose answer is known by construction.
- Invariance to the internal chunking used to bound memory.
- The ``ia-spa`` command-line interface, end to end.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ia_spa import (
    RadioMapSet,
    TowerOptimizer,
    greedy_select,
    load_locations,
    load_radio_maps,
    place_towers,
    run_greedy,
    w_bar,
)
from ia_spa.cli import main as cli_main

C = 1e8  # saturation constant used throughout the tests


# ===========================================================================
# Synthetic problem generators
# ===========================================================================

def make_problem(n_sites=12, height=24, width=24, seed=0):
    """Distance-decayed rate maps on a grid of candidate sites."""
    rng = np.random.default_rng(seed)
    sites_xy = rng.uniform(2, min(height, width) - 2, size=(n_sites, 2))
    sites = np.column_stack([sites_xy, np.full(n_sites, 15.0)])

    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    maps = np.empty((n_sites, height, width))
    for i, (x, y, _) in enumerate(sites):
        distance = np.hypot(rows - x, cols - y) + 1.0
        maps[i] = 4e9 / distance ** 2
    return maps, sites


def make_disjoint_problem(block_sizes=(9, 6, 4, 2), level=5e8):
    """Maps that each cover one disjoint block, of known decreasing value.

    The greedy optimum is unambiguous: the blocks in decreasing size order.
    """
    n = len(block_sizes)
    width = sum(block_sizes)
    maps = np.zeros((n, 1, width))
    start = 0
    for i, size in enumerate(block_sizes):
        maps[i, 0, start:start + size] = level
        start += size
    sites = np.column_stack([np.arange(n), np.zeros(n), np.zeros(n)]).astype(float)
    return maps, sites


# ===========================================================================
# Reference implementation -- deliberately naive, for cross-checking
# ===========================================================================

def reference_quality(maps, chosen, aggregation="max", c=C, density=None):
    """S(T) computed the slow, obvious way."""
    field = np.zeros(maps.shape[1:])
    for i in chosen:
        field = np.maximum(field, maps[i]) if aggregation == "max" else field + maps[i]
    utility = w_bar(field, c)
    if density is None:
        return float(utility.mean())
    weights = np.asarray(density, dtype=float)
    return float((utility * weights / weights.sum()).sum())


def reference_greedy(maps, k, aggregation="max", c=C, density=None, fixed=()):
    """Greedy selection written out longhand, without any optimisation."""
    chosen = list(fixed)
    picked = []
    for _ in range(k):
        base = reference_quality(maps, chosen, aggregation, c, density)
        gains = [
            reference_quality(maps, chosen + [i], aggregation, c, density) - base
            for i in range(len(maps))
        ]
        for i in chosen:
            gains[i] = -np.inf  # 'max' aggregation forbids repeats
        best = int(np.argmax(gains))
        chosen.append(best)
        picked.append(best)
    return picked


# ===========================================================================
# 1. Loading radio maps
# ===========================================================================

class TestLoadRadioMaps:
    def test_npy_stack(self, tmp_path):
        maps, _ = make_problem()
        np.save(tmp_path / "maps.npy", maps)
        np.testing.assert_allclose(load_radio_maps(tmp_path / "maps.npy"), maps)

    def test_npy_is_memory_mapped(self, tmp_path):
        maps, _ = make_problem()
        np.save(tmp_path / "maps.npy", maps)
        assert isinstance(load_radio_maps(tmp_path / "maps.npy"), np.memmap)
        assert not isinstance(
            load_radio_maps(tmp_path / "maps.npy", mmap=False), np.memmap
        )

    def test_npz_single_array(self, tmp_path):
        maps, _ = make_problem()
        np.savez(tmp_path / "maps.npz", radio_maps=maps)
        np.testing.assert_allclose(load_radio_maps(tmp_path / "maps.npz"), maps)

    def test_npz_one_array_per_candidate_in_numeric_order(self, tmp_path):
        maps, _ = make_problem(n_sites=12)
        np.savez(tmp_path / "maps.npz", **{str(i): m for i, m in enumerate(maps)})
        loaded = load_radio_maps(tmp_path / "maps.npz")
        # Naive string sorting would put '10' before '2'.
        np.testing.assert_allclose(loaded, maps)

    def test_text_table_of_flattened_maps(self, tmp_path):
        maps, _ = make_problem(height=8, width=8)
        np.savetxt(tmp_path / "maps.csv", maps.reshape(len(maps), -1), delimiter=",")
        loaded = load_radio_maps(tmp_path / "maps.csv")
        np.testing.assert_allclose(loaded, maps.reshape(len(maps), -1))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_radio_maps(tmp_path / "nope.npy")

    def test_directory_rejected_by_single_file_loader(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            load_radio_maps(tmp_path)

    def test_unknown_suffix_raises(self, tmp_path):
        (tmp_path / "maps.bin").write_bytes(b"\x00")
        with pytest.raises(ValueError, match="Unsupported radio-map format"):
            load_radio_maps(tmp_path / "maps.bin")

    def test_npz_mismatched_shapes_raises(self, tmp_path):
        np.savez(tmp_path / "m.npz", **{"0": np.ones((4, 4)), "1": np.ones((5, 5))})
        with pytest.raises(ValueError, match="differing shapes"):
            load_radio_maps(tmp_path / "m.npz")


# ===========================================================================
# 2. Loading locations
# ===========================================================================

class TestLoadLocations:
    def test_npy(self, tmp_path):
        _, sites = make_problem()
        np.save(tmp_path / "s.npy", sites)
        np.testing.assert_allclose(load_locations(tmp_path / "s.npy"), sites)

    def test_csv_with_header(self, tmp_path):
        _, sites = make_problem()
        np.savetxt(tmp_path / "s.csv", sites, delimiter=",",
                   header="x,y,z", comments="")
        np.testing.assert_allclose(load_locations(tmp_path / "s.csv"), sites)

    def test_whitespace_txt(self, tmp_path):
        _, sites = make_problem()
        np.savetxt(tmp_path / "s.txt", sites)
        np.testing.assert_allclose(load_locations(tmp_path / "s.txt"), sites)

    def test_index_column_is_dropped(self, tmp_path):
        """The `i, x, y, z` manifest written by the Sionna pipeline."""
        _, sites = make_problem(n_sites=5)
        with open(tmp_path / "0_Coordinates.txt", "w") as fh:
            for i, (x, y, z) in enumerate(sites):
                fh.write(f"{i}, {x}, {y}, {z}\n")
        np.testing.assert_allclose(load_locations(tmp_path / "0_Coordinates.txt"), sites)

    def test_xy_only_gets_zero_height(self, tmp_path):
        np.savetxt(tmp_path / "s.csv", np.array([[1.0, 2.0], [3.0, 4.0]]), delimiter=",")
        with pytest.warns(UserWarning, match="z is set to 0"):
            coords = load_locations(tmp_path / "s.csv")
        np.testing.assert_allclose(coords, [[1, 2, 0], [3, 4, 0]])

    def test_four_columns_without_index_raises(self, tmp_path):
        np.savetxt(tmp_path / "s.csv", np.ones((3, 4)), delimiter=",")
        with pytest.raises(ValueError, match="index column"):
            load_locations(tmp_path / "s.csv")

    def test_too_many_columns_raises(self, tmp_path):
        np.savetxt(tmp_path / "s.csv", np.ones((3, 5)), delimiter=",")
        with pytest.raises(ValueError, match="expected 2, 3 or 4 columns"):
            load_locations(tmp_path / "s.csv")

    def test_nan_coordinates_raise(self, tmp_path):
        sites = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0]])
        np.save(tmp_path / "s.npy", sites)
        with pytest.raises(ValueError, match="NaN"):
            load_locations(tmp_path / "s.npy")


# ===========================================================================
# 3. RadioMapSet validation
# ===========================================================================

class TestRadioMapSetValidation:
    def test_count_mismatch_raises(self):
        maps, sites = make_problem(n_sites=6)
        with pytest.raises(ValueError, match="1:1 order"):
            RadioMapSet(maps=maps, locations=sites[:4])

    def test_negative_values_rejected_with_db_hint(self):
        """Handing over path gain in dB is a plausible mistake; catch it."""
        _, sites = make_problem(n_sites=4)
        path_gain = np.full((4, 8, 8), 1e-9)          # linear, as required
        maps_db = 10 * np.log10(path_gain)            # -90 dB everywhere
        with pytest.raises(ValueError, match="dB"):
            RadioMapSet(maps=maps_db, locations=sites)
        RadioMapSet(maps=path_gain, locations=sites)  # the linear form is fine

    def test_nan_map_rejected(self):
        maps, sites = make_problem(n_sites=4)
        maps[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            RadioMapSet(maps=maps, locations=sites)

    def test_directory_layout_round_trip(self, tmp_path):
        maps, sites = make_problem(n_sites=5)
        sionna = tmp_path / "Sionna"
        sionna.mkdir()
        for i, m in enumerate(maps):
            np.save(sionna / f"{i}.npy", m)
        with open(sionna / "0_Coordinates.txt", "w") as fh:
            for i, (x, y, z) in enumerate(sites):
                fh.write(f"{i}, {x}, {y}, {z}\n")

        maps_set = RadioMapSet.from_files(tmp_path)
        assert maps_set.n_maps == 5
        np.testing.assert_allclose(maps_set.locations, sites)
        np.testing.assert_allclose(maps_set.map_at(3), maps[3])

    def test_directory_with_gaps_raises(self, tmp_path):
        for i in (0, 1, 3):
            np.save(tmp_path / f"{i}.npy", np.ones((4, 4)))
        with pytest.raises(ValueError, match="not numbered"):
            RadioMapSet.from_files(tmp_path, np.zeros((3, 3)))

    def test_directory_without_maps_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No numbered radio-map files"):
            RadioMapSet.from_files(tmp_path, np.zeros((3, 3)))

    def test_ragged_directory_maps_raise(self, tmp_path):
        np.save(tmp_path / "0.npy", np.ones((4, 4)))
        np.save(tmp_path / "1.npy", np.ones((5, 5)))
        with pytest.raises(ValueError, match="All maps must share one grid"):
            RadioMapSet.from_files(tmp_path, np.zeros((2, 3)))

    def test_stacked_file_without_locations_raises(self, tmp_path):
        maps, _ = make_problem(n_sites=3)
        np.save(tmp_path / "maps.npy", maps)
        with pytest.raises(ValueError, match="locations file is required"):
            RadioMapSet.from_files(tmp_path / "maps.npy")

    def test_one_dimensional_maps_raise(self):
        with pytest.raises(ValueError, match=r"expected \(N, H, W\)"):
            RadioMapSet(maps=np.arange(6.0), locations=np.zeros((6, 3)))

    def test_flattened_maps_are_accepted(self):
        maps, sites = make_problem(n_sites=6, height=10, width=10)
        flat = RadioMapSet(maps=maps.reshape(6, -1), locations=sites)
        assert flat.map_shape == (100,)
        assert flat.n_cells == 100


# ===========================================================================
# 4. Accuracy: cross-check against a naive reference implementation
# ===========================================================================

class TestAgainstReference:
    @pytest.mark.parametrize("aggregation", ["max", "sum"])
    def test_marginal_gains_match_definition(self, aggregation):
        """G(x|T) must equal S(T u {x}) - S(T) evaluated from scratch."""
        maps, sites = make_problem(n_sites=10, seed=1)
        opt = TowerOptimizer(maps, locations=sites, aggregation=aggregation)

        u = np.zeros(len(maps))
        u[[2, 5]] = 1
        gains = opt.marginal_gains(u)

        base = reference_quality(maps, [2, 5], aggregation)
        expected = [
            reference_quality(maps, [2, 5, i], aggregation) - base
            for i in range(len(maps))
        ]
        np.testing.assert_allclose(gains, expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("aggregation", ["max", "sum"])
    def test_selection_matches_naive_greedy(self, aggregation):
        maps, sites = make_problem(n_sites=14, seed=2)
        result = greedy_select(
            TowerOptimizer(maps, locations=sites, aggregation=aggregation),
            n_towers=5, allow_repeats=False, progress=False, verbose=False,
        )
        assert list(result.indices) == reference_greedy(maps, 5, aggregation)

    def test_quality_trajectory_matches_reference(self):
        maps, sites = make_problem(n_sites=12, seed=3)
        result = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=4, progress=False, verbose=False,
        )
        expected = [
            reference_quality(maps, list(result.indices[:k]))
            for k in range(len(result.indices) + 1)
        ]
        np.testing.assert_allclose(result.quality, expected, rtol=1e-12, atol=1e-15)

    def test_warm_start_matches_reference(self):
        maps, sites = make_problem(n_sites=12, seed=4)
        opt = TowerOptimizer(maps, locations=sites)
        result = greedy_select(
            opt, n_towers=3, fixed_towers=sites[[1, 7]],
            progress=False, verbose=False,
        )
        assert list(result.fixed_indices) == [1, 7]
        assert list(result.indices) == reference_greedy(maps, 3, fixed=[1, 7])

    def test_density_weighting_matches_reference(self):
        maps, sites = make_problem(n_sites=10, seed=5)
        rng = np.random.default_rng(11)
        density = rng.uniform(0.0, 1.0, size=maps.shape[1:])

        opt = TowerOptimizer(maps, locations=sites, density=density)
        result = greedy_select(opt, n_towers=3, progress=False, verbose=False)
        assert list(result.indices) == reference_greedy(maps, 3, density=density)
        assert result.quality[-1] == pytest.approx(
            reference_quality(maps, list(result.indices), density=density)
        )

    def test_density_is_normalised(self):
        """Scaling the density must not change S(T): it is an expectation."""
        maps, sites = make_problem(n_sites=8, seed=6)
        density = np.full(maps.shape[1:], 3.7)
        u = np.zeros(len(maps))
        u[[0, 4]] = 1
        weighted = TowerOptimizer(maps, locations=sites, density=density)
        uniform = TowerOptimizer(maps, locations=sites)
        assert weighted.network_quality(u) == pytest.approx(uniform.network_quality(u))


# ===========================================================================
# 5. Accuracy: exhaustive optimum and the submodular guarantee
# ===========================================================================

class TestAgainstBruteForce:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_greedy_meets_one_minus_one_over_e_bound(self, seed):
        """Theorem III.2: S(T_greedy) >= (1 - 1/e) S(T*) for equal budget."""
        maps, sites = make_problem(n_sites=9, height=14, width=14, seed=seed)
        k = 3
        result = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=k, progress=False, verbose=False,
        )
        best = max(
            reference_quality(maps, list(subset))
            for subset in itertools.combinations(range(len(maps)), k)
        )
        bound = (1 - np.exp(-1)) * best
        assert result.quality[-1] >= bound
        # In practice greedy is near-optimal, far above the worst-case bound.
        assert result.quality[-1] >= 0.95 * best

    def test_first_pick_is_the_single_best_site(self):
        maps, sites = make_problem(n_sites=15, seed=7)
        result = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=1, progress=False, verbose=False,
        )
        best = int(np.argmax([reference_quality(maps, [i]) for i in range(len(maps))]))
        assert result.indices[0] == best

    def test_known_answer_disjoint_blocks(self):
        """Disjoint coverage blocks must be picked in decreasing size order."""
        maps, sites = make_disjoint_problem(block_sizes=(9, 6, 4, 2))
        result = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=4, progress=False, verbose=False,
        )
        assert list(result.indices) == [0, 1, 2, 3]
        # Every gain is proportional to the block it covers.
        ratios = result.gains / np.array([9, 6, 4, 2])
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-12)

    def test_diminishing_returns(self):
        maps, sites = make_problem(n_sites=12, seed=8)
        result = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=6, progress=False, verbose=False,
        )
        assert np.all(np.diff(result.gains) <= 1e-12)

    def test_submodularity_of_s(self):
        """G(x|A) >= G(x|B) whenever A is a subset of B."""
        maps, sites = make_problem(n_sites=10, seed=9)
        opt = TowerOptimizer(maps, locations=sites)
        u_small = np.zeros(len(maps)); u_small[[1]] = 1
        u_large = u_small.copy(); u_large[[3, 6]] = 1
        assert np.all(opt.marginal_gains(u_small) >= opt.marginal_gains(u_large) - 1e-12)


# ===========================================================================
# 6. Greedy behaviour and bookkeeping
# ===========================================================================

class TestGreedyBehaviour:
    def test_no_duplicate_sites_under_max(self):
        maps, sites = make_disjoint_problem(block_sizes=(6, 3))
        with pytest.warns(UserWarning, match="already in use"):
            result = greedy_select(
                TowerOptimizer(maps, locations=sites),
                n_towers=5, progress=False, verbose=False,
            )
        assert list(result.indices) == [0, 1]
        assert len(set(result.indices)) == len(result.indices)

    def test_stops_when_no_gain_remains(self):
        maps, sites = make_problem(n_sites=4, seed=10)
        maps[2:] = 0.0  # two useless candidates
        with pytest.warns(UserWarning, match="no further transmitter improves"):
            result = greedy_select(
                TowerOptimizer(maps, locations=sites),
                n_towers=4, progress=False, verbose=False,
            )
        assert result.n_towers == 2

    def test_repeats_allowed_under_sum(self):
        """P_SUM may re-use a site, since co-located power does add up."""
        maps = np.zeros((2, 1, 4))
        maps[0, 0, :] = 1e9     # strong, wide
        maps[1, 0, 0] = 1.0     # negligible
        sites = np.array([[0.0, 0, 0], [1.0, 0, 0]])
        result = greedy_select(
            TowerOptimizer(maps, locations=sites, aggregation="sum"),
            n_towers=2, progress=False, verbose=False,
        )
        assert list(result.indices) == [0, 0]

    def test_u_vector_counts_all_transmitters(self):
        maps, sites = make_problem(n_sites=10, seed=12)
        result = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=3, fixed_towers=sites[[0]], progress=False, verbose=False,
        )
        assert result.u.sum() == pytest.approx(4)
        assert result.all_positions.shape == (4, 3)

    def test_positions_correspond_to_indices(self):
        maps, sites = make_problem(n_sites=10, seed=13)
        result = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=4, progress=False, verbose=False,
        )
        np.testing.assert_allclose(result.positions, sites[result.indices])

    def test_zero_towers_is_a_no_op(self):
        maps, sites = make_problem(n_sites=6, seed=14)
        result = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=0, progress=False, verbose=False,
        )
        assert result.n_towers == 0
        assert result.positions.shape == (0, 3)
        assert result.quality.tolist() == [0.0]

    def test_negative_n_towers_raises(self):
        maps, sites = make_problem(n_sites=6, seed=15)
        with pytest.raises(ValueError, match="non-negative"):
            greedy_select(TowerOptimizer(maps, locations=sites), n_towers=-1)

    def test_warm_start_snaps_to_nearest_candidate(self):
        maps, sites = make_problem(n_sites=8, seed=16)
        opt = TowerOptimizer(maps, locations=sites)
        nudged = sites[[3, 5]] + np.array([0.01, -0.01, 0.0])
        assert list(opt.snap_to_candidates(nudged)) == [3, 5]

    @pytest.mark.parametrize("chunk_bytes", [1, 1024, 1 << 20])
    def test_chunking_does_not_change_results(self, chunk_bytes):
        """Memory chunking is an implementation detail, not a result change."""
        maps, sites = make_problem(n_sites=11, seed=17)
        reference = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=4, progress=False, verbose=False,
        )
        chunked = greedy_select(
            TowerOptimizer(maps, locations=sites, chunk_bytes=chunk_bytes),
            n_towers=4, progress=False, verbose=False,
        )
        assert list(chunked.indices) == list(reference.indices)
        np.testing.assert_allclose(chunked.gains, reference.gains, rtol=1e-12)

    def test_input_arrays_are_not_modified(self):
        maps, sites = make_problem(n_sites=8, seed=18)
        maps_before = maps.copy()
        greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=3, progress=False, verbose=False,
        )
        np.testing.assert_array_equal(maps, maps_before)


# ===========================================================================
# 6b. Subclass hooks -- the extension point the notebooks rely on
# ===========================================================================

class MaskedTowerOptimizer(TowerOptimizer):
    """The exclusion-zone pattern from notebooks/03_florence_advanced.ipynb."""

    def __init__(self, *args, exclusion_mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.exclusion_mask = exclusion_mask  # True -> forbidden

    def marginal_gains(self, u, density=None):
        gains = super().marginal_gains(u, density)
        gains[self.exclusion_mask] = 0.0
        return gains


class TestSubclassHooks:
    def test_overridden_marginal_gains_is_honoured(self):
        """A subclass that masks candidates must actually exclude them."""
        maps, sites = make_problem(n_sites=12, seed=40)
        allowed = [2, 5, 7]
        mask = np.ones(len(maps), dtype=bool)
        mask[allowed] = False

        masked = greedy_select(
            MaskedTowerOptimizer(maps, locations=sites, exclusion_mask=mask),
            n_towers=3, progress=False, verbose=False,
        )
        assert sorted(masked.indices) == allowed

        # The order must match greedy run on the allowed sites alone.
        restricted = greedy_select(
            TowerOptimizer(maps[allowed], locations=sites[allowed]),
            n_towers=3, progress=False, verbose=False,
        )
        assert list(masked.indices) == [allowed[i] for i in restricted.indices]

    def test_unconstrained_run_ignores_the_mask_hook(self):
        """Sanity check: without a mask the same subclass changes nothing."""
        maps, sites = make_problem(n_sites=12, seed=40)
        plain = greedy_select(
            TowerOptimizer(maps, locations=sites),
            n_towers=3, progress=False, verbose=False,
        )
        subclassed = greedy_select(
            MaskedTowerOptimizer(
                maps, locations=sites, exclusion_mask=np.zeros(len(maps), bool)
            ),
            n_towers=3, progress=False, verbose=False,
        )
        assert list(plain.indices) == list(subclassed.indices)

    def test_override_works_through_run_greedy(self, tmp_path):
        maps, sites = make_problem(n_sites=10, seed=41)
        mask = np.zeros(len(maps), dtype=bool)
        mask[[0, 1, 2, 3, 4]] = True
        run_greedy(
            MaskedTowerOptimizer(maps, locations=sites, exclusion_mask=mask),
            results_folder=tmp_path / "r", n_iter=3, progress=False, verbose=False,
        )
        chosen = np.loadtxt(tmp_path / "r" / "towers.csv", delimiter=",",
                            skiprows=1)[:, 0].astype(int)
        assert not set(chosen) & {0, 1, 2, 3, 4}

    def test_bad_override_shape_is_reported(self):
        class Broken(TowerOptimizer):
            def marginal_gains(self, u, density=None):
                return np.zeros(3)

        maps, sites = make_problem(n_sites=8, seed=42)
        with pytest.raises(ValueError, match="marginal_gains returned shape"):
            greedy_select(Broken(maps, locations=sites), n_towers=1,
                          progress=False, verbose=False)


# ===========================================================================
# 7. place_towers -- the high-level entry point
# ===========================================================================

class TestPlaceTowers:
    def test_more_towers_than_candidates_stops_early(self):
        maps, sites = make_problem(n_sites=3, seed=28)
        with pytest.warns(UserWarning, match="already in use"):
            result = place_towers(maps, sites, n_towers=10,
                                  progress=False, verbose=False)
        assert result.n_towers == 3

    def test_single_candidate(self):
        maps, sites = make_problem(n_sites=1, seed=29)
        result = place_towers(maps, sites, n_towers=1, progress=False, verbose=False)
        np.testing.assert_allclose(result.positions, sites)

    def test_arrays_in_result_out(self):
        maps, sites = make_problem(n_sites=12, seed=19)
        result = place_towers(maps, sites, n_towers=4, progress=False, verbose=False)
        assert result.positions.shape == (4, 3)
        np.testing.assert_allclose(result.positions, sites[result.indices])

    def test_files_in_file_out(self, tmp_path):
        maps, sites = make_problem(n_sites=12, seed=20)
        np.save(tmp_path / "maps.npy", maps)
        np.savetxt(tmp_path / "sites.csv", sites, delimiter=",",
                   header="x,y,z", comments="")

        result = place_towers(
            tmp_path / "maps.npy", tmp_path / "sites.csv", n_towers=5,
            output=tmp_path / "towers.csv", progress=False, verbose=False,
        )
        written = np.loadtxt(tmp_path / "towers.csv", delimiter=",", skiprows=1)
        np.testing.assert_allclose(written[:, 1:4], result.positions, atol=1e-6)
        np.testing.assert_array_equal(written[:, 0].astype(int), result.indices)

    def test_output_directory_gets_summary(self, tmp_path):
        maps, sites = make_problem(n_sites=8, seed=21)
        place_towers(maps, sites, n_towers=3, output=tmp_path / "run",
                     progress=False, verbose=False)
        assert (tmp_path / "run" / "towers.csv").exists()
        assert (tmp_path / "run" / "summary.json").exists()

    def test_output_npy(self, tmp_path):
        maps, sites = make_problem(n_sites=8, seed=22)
        result = place_towers(maps, sites, n_towers=3, output=tmp_path / "t.npy",
                              progress=False, verbose=False)
        np.testing.assert_allclose(np.load(tmp_path / "t.npy"), result.positions)

    def test_directory_of_maps_needs_no_locations_argument(self, tmp_path):
        maps, sites = make_problem(n_sites=6, seed=23)
        for i, m in enumerate(maps):
            np.save(tmp_path / f"{i}.npy", m)
        with open(tmp_path / "0_Coordinates.txt", "w") as fh:
            for i, (x, y, z) in enumerate(sites):
                fh.write(f"{i}, {x}, {y}, {z}\n")
        result = place_towers(tmp_path, n_towers=3, progress=False, verbose=False)
        assert result.positions.shape == (3, 3)

    def test_missing_locations_raises(self):
        maps, _ = make_problem(n_sites=6, seed=24)
        with pytest.raises(ValueError, match="'locations' is required"):
            place_towers(maps, None, n_towers=2, verbose=False)

    def test_utility_scale_warning_when_maps_far_below_c(self):
        maps, sites = make_problem(n_sites=6, seed=25)
        with pytest.warns(UserWarning, match="effectively linear"):
            place_towers(maps * 1e-9, sites, n_towers=1,
                         progress=False, verbose=False)

    def test_utility_scale_warning_when_maps_far_above_c(self):
        maps, sites = make_problem(n_sites=6, seed=26)
        with pytest.warns(UserWarning, match="saturated"):
            place_towers(maps * 1e6, sites, n_towers=1,
                         progress=False, verbose=False)

    def test_summary_is_printable(self):
        maps, sites = make_problem(n_sites=8, seed=27)
        result = place_towers(maps, sites, n_towers=3, progress=False, verbose=False)
        text = result.summary()
        assert "IA-SPA placement" in text
        assert text.count("\n") >= 3 + 3


# ===========================================================================
# 8. Command-line interface
# ===========================================================================

@pytest.fixture()
def cli_inputs(tmp_path):
    maps, sites = make_problem(n_sites=12, seed=30)
    np.save(tmp_path / "maps.npy", maps)
    np.savetxt(tmp_path / "sites.csv", sites, delimiter=",",
               header="x,y,z", comments="")
    return tmp_path, maps, sites


class TestCLI:
    def test_end_to_end(self, cli_inputs, capsys):
        tmp_path, maps, sites = cli_inputs
        code = cli_main([
            "-m", str(tmp_path / "maps.npy"),
            "-l", str(tmp_path / "sites.csv"),
            "-n", "4",
            "-o", str(tmp_path / "towers.csv"),
            "-q",
        ])
        assert code == 0
        written = np.loadtxt(tmp_path / "towers.csv", delimiter=",", skiprows=1)
        assert written.shape == (4, 5)
        assert list(written[:, 0].astype(int)) == reference_greedy(maps, 4)
        assert "IA-SPA placement" in capsys.readouterr().out

    def test_check_only_does_not_write_output(self, cli_inputs, capsys):
        tmp_path, _, _ = cli_inputs
        code = cli_main([
            "-m", str(tmp_path / "maps.npy"),
            "-l", str(tmp_path / "sites.csv"),
            "--check-only",
        ])
        assert code == 0
        assert not (tmp_path / "towers.csv").exists()
        assert "Inputs are valid" in capsys.readouterr().out

    def test_missing_file_reports_error(self, tmp_path, capsys):
        code = cli_main(["-m", str(tmp_path / "absent.npy"), "-l", str(tmp_path)])
        assert code == 2
        assert "ERROR" in capsys.readouterr().err

    def test_mismatched_counts_report_error(self, tmp_path, capsys):
        maps, sites = make_problem(n_sites=6, seed=31)
        np.save(tmp_path / "maps.npy", maps)
        np.savetxt(tmp_path / "sites.csv", sites[:3], delimiter=",")
        code = cli_main([
            "-m", str(tmp_path / "maps.npy"), "-l", str(tmp_path / "sites.csv"),
        ])
        assert code == 2
        assert "1:1 order" in capsys.readouterr().err

    def test_sum_aggregation_flag(self, cli_inputs):
        tmp_path, maps, _ = cli_inputs
        code = cli_main([
            "-m", str(tmp_path / "maps.npy"), "-l", str(tmp_path / "sites.csv"),
            "-n", "3", "-a", "sum", "--no-repeats",
            "-o", str(tmp_path / "sum.csv"), "-q",
        ])
        assert code == 0
        written = np.loadtxt(tmp_path / "sum.csv", delimiter=",", skiprows=1)
        assert list(written[:, 0].astype(int)) == reference_greedy(maps, 3, "sum")

    def test_runs_as_a_module(self, cli_inputs):
        """`python -m ia_spa` must work without the console script installed."""
        tmp_path, _, _ = cli_inputs
        proc = subprocess.run(
            [sys.executable, "-m", "ia_spa",
             "-m", str(tmp_path / "maps.npy"),
             "-l", str(tmp_path / "sites.csv"),
             "-n", "2", "-o", str(tmp_path / "m.csv"), "-q"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert (tmp_path / "m.csv").exists()

    def test_warm_start_and_density_flags(self, cli_inputs):
        tmp_path, maps, sites = cli_inputs
        np.savetxt(tmp_path / "fixed.csv", sites[[0, 1]], delimiter=",")
        np.save(tmp_path / "density.npy",
                np.linspace(0.1, 1.0, maps[0].size).reshape(maps[0].shape))

        code = cli_main([
            "-m", str(tmp_path / "maps.npy"), "-l", str(tmp_path / "sites.csv"),
            "-n", "3",
            "--fixed-towers", str(tmp_path / "fixed.csv"),
            "--density", str(tmp_path / "density.npy"),
            "-o", str(tmp_path / "warm.csv"), "-q",
        ])
        assert code == 0
        chosen = np.loadtxt(tmp_path / "warm.csv", delimiter=",",
                            skiprows=1)[:, 0].astype(int)
        assert len(chosen) == 3
        # The two warm-start sites are already served, so they are not re-picked.
        assert not set(chosen) & {0, 1}

    def test_txt_output_round_trips(self, cli_inputs):
        tmp_path, _, _ = cli_inputs
        code = cli_main([
            "-m", str(tmp_path / "maps.npy"), "-l", str(tmp_path / "sites.csv"),
            "-n", "3", "-o", str(tmp_path / "towers.txt"), "-q",
        ])
        assert code == 0
        # A bare `x, y, z` file, readable by the Sionna-side evaluation script.
        assert load_locations(tmp_path / "towers.txt").shape == (3, 3)

    def test_help_mentions_the_inputs(self, capsys):
        with pytest.raises(SystemExit):
            cli_main(["--help"])
        out = capsys.readouterr().out
        assert "--radio-maps" in out and "--locations" in out
