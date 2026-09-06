"""
ia_spa/optimizer.py
-------------------
Core implementation of the Interference-Aware Submodular Placement Algorithm (IA-SPA).

This module provides the :class:`TowerOptimizer` class, which encapsulates the
aggregated network quality functional S(T), and the greedy placement loop
described in:

    L. Taus, R. Tsai, and J. G. Andrews, "Optimal Transmitter Placement in
    Realistic Urban Environments," submitted to IEEE Transactions on Wireless
    Communications, arXiv:2604.28153 [cs.IT], Apr. 2026.
    https://arxiv.org/abs/2604.28153

The key algorithmic idea is to iteratively select the candidate transmitter
location that maximises the marginal gain G(x|T) = S(T u {x}) - S(T), where
S(T) is the expectation of the concave utility W applied to the aggregated
path-gain field.

Usage
-----
    from ia_spa import TowerOptimizer, greedy_select

    # From a stacked radio-map array and its candidate coordinates
    optimizer = TowerOptimizer(radio_maps, locations=coords, aggregation="max")
    result = greedy_select(optimizer, n_towers=20)
    print(result.positions)

    # From the folder layout written by scripts/compute_basis.py
    optimizer = TowerOptimizer("data/BasisFunctions/SF", aggregation="max")
    run_greedy(optimizer, results_folder="data/Results/SF_max", n_iter=20)
"""

from __future__ import annotations

import inspect
import json
import warnings
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from ia_spa.data import (
    DEFAULT_CHUNK_BYTES,
    RadioMapSet,
    _normalise_locations,
    coerce_locations,
    save_positions,
)

__all__ = [
    "PlacementResult",
    "TowerOptimizer",
    "greedy_select",
    "run_greedy",
    "w_bar",
]

PathLike = Union[str, Path]

# Below this many radio-map values per scoring pass, the scan is fast enough
# that a progress bar is pure noise.
_PROGRESS_MIN_WORK = 20_000_000


# ---------------------------------------------------------------------------
# Utility functional
# ---------------------------------------------------------------------------

def w_bar(x: np.ndarray, c: float = 1e8) -> np.ndarray:
    """Concave utility function W(x) = x / (x + c).

    This satisfies the assumptions of Theorem II.4 in the paper: W is
    non-negative, non-decreasing, and concave, which guarantees submodularity
    of S(T) and the greedy approximation bound of Theorem III.2.

    Parameters
    ----------
    x : np.ndarray
        Array of aggregated path-gain (or rate) values.  Must be non-negative;
        the function has a pole at x = -c.
    c : float, optional
        Saturation constant. Default 1e8 corresponds to 100 Mbps normalisation
        when x is expressed in bps.

    Returns
    -------
    np.ndarray
        Element-wise utility values, in [0, 1).
    """
    return x / (x + c)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PlacementResult:
    """Outcome of a greedy IA-SPA run.

    Attributes
    ----------
    positions : np.ndarray, shape (k, 3)
        Coordinates of the newly placed transmitters, in selection order.
    indices : np.ndarray, shape (k,)
        Index into the candidate list for each placed transmitter.
    gains : np.ndarray, shape (k,)
        Marginal gain G(x|T) realised at each step.
    quality : np.ndarray, shape (k + 1,)
        Network quality S(T) before the first placement and after each step.
    fixed_positions : np.ndarray, shape (m, 3)
        Warm-start transmitters, snapped to the nearest candidate sites.
    fixed_indices : np.ndarray, shape (m,)
        Candidate indices of the warm-start transmitters.
    u : np.ndarray, shape (n_candidates,)
        Final weight vector over all candidates (fixed + placed).
    aggregation, utility_c, n_candidates :
        The settings the run used, kept for reproducibility.
    """

    positions: np.ndarray
    indices: np.ndarray
    gains: np.ndarray
    quality: np.ndarray
    u: np.ndarray
    fixed_positions: np.ndarray = dataclass_field(
        default_factory=lambda: np.empty((0, 3))
    )
    fixed_indices: np.ndarray = dataclass_field(
        default_factory=lambda: np.empty(0, dtype=int)
    )
    aggregation: str = "max"
    utility_c: float = 1e8
    n_candidates: int = 0

    @property
    def n_towers(self) -> int:
        """Number of newly placed transmitters."""
        return len(self.positions)

    @property
    def all_positions(self) -> np.ndarray:
        """Warm-start transmitters followed by the newly placed ones."""
        return np.vstack([self.fixed_positions, self.positions])

    def summary(self) -> str:
        """Human-readable report of the run."""
        lines = [
            f"IA-SPA placement: {self.n_towers} transmitter(s) chosen from "
            f"{self.n_candidates} candidate site(s)",
            f"  aggregation = {self.aggregation}   utility_c = {self.utility_c:g}",
        ]
        if len(self.fixed_indices):
            lines.append(
                f"  warm start  = {len(self.fixed_indices)} pre-placed transmitter(s)"
            )
        lines.append(
            f"  S(T) = {self.quality[0]:.6f} -> {self.quality[-1]:.6f} "
            f"(+{self.quality[-1] - self.quality[0]:.6f})"
        )
        lines.append("  #   candidate            x           y           z       gain")
        for step, (idx, (x, y, z), gain) in enumerate(
            zip(self.indices, self.positions, self.gains), start=1
        ):
            lines.append(
                f"  {step:<3d} {idx:<9d} {x:11.2f} {y:11.2f} {z:11.2f} {gain:10.6f}"
            )
        return "\n".join(lines)

    def save(self, path: PathLike) -> Path:
        """Save the placed tower positions.

        If *path* has no suffix it is treated as a directory and
        ``towers.csv`` plus ``summary.json`` are written inside it;
        otherwise the positions are written to that single file (``.csv``,
        ``.txt`` or ``.npy``).

        Returns
        -------
        Path
            The file containing the tower positions.
        """
        path = Path(path)
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            target = path / "towers.csv"
            (path / "summary.json").write_text(json.dumps(self.to_dict(), indent=2))
        else:
            target = path
        return save_positions(target, self.positions, self.indices, self.gains)

    def to_dict(self) -> dict:
        """JSON-serialisable view of the run, for provenance and reporting."""
        return {
            "aggregation": self.aggregation,
            "utility_c": self.utility_c,
            "n_candidates": int(self.n_candidates),
            "n_towers": int(self.n_towers),
            "indices": [int(i) for i in self.indices],
            "positions": [[float(v) for v in p] for p in self.positions],
            "gains": [float(g) for g in self.gains],
            "quality": [float(s) for s in self.quality],
            "fixed_indices": [int(i) for i in self.fixed_indices],
        }


# ---------------------------------------------------------------------------
# Main optimiser class
# ---------------------------------------------------------------------------

class TowerOptimizer:
    """Evaluates S(T) and the marginal gain G(x|T) over pre-computed radio maps.

    Parameters
    ----------
    radio_maps : RadioMapSet, np.ndarray, str or Path
        The per-candidate radio maps, given as

        * a :class:`~ia_spa.data.RadioMapSet`,
        * an ``(N, H, W)`` array (then *locations* is required),
        * a path to a stacked ``.npy`` / ``.npz`` / text file (then
          *locations* is required), or
        * a path to a basis-function folder written by
          ``scripts/compute_basis.py`` -- the folder's ``Sionna/`` sub-directory
          and its ``0_Coordinates.txt`` manifest are found automatically.
    aggregation : {"max", "sum"}
        Aggregation rule for the multi-transmitter field P(y, T):

        * ``"max"``  - P_MAX(y, T) = max over t in T of P(y, t).  Corresponds
          to a user associating with the strongest BS; the realistic default.
        * ``"sum"``  - P_SUM(y, T) = sum over t in T of P(y, t).  Idealistic
          upper bound that treats all signals as constructive.
    utility_c : float, optional
        Saturation constant for W.  See :func:`w_bar`.
    locations : np.ndarray or path, optional
        Candidate (x, y, z) coordinates, one per radio map and in the same
        order.  Required unless *radio_maps* already carries them.
    density : np.ndarray, optional
        Spatial priority density f(y) over the map cells.  Normalised to sum
        to one.  Defaults to uniform.
    basis_folder : str or Path, optional
        Backwards-compatible alias for passing a basis-function folder.
    chunk_bytes : int, optional
        Radio-map bytes read at once while scoring candidates.

    Raises
    ------
    ValueError
        For an unknown aggregation rule, or missing / mismatched locations.
    FileNotFoundError
        If the radio maps or the coordinate manifest cannot be found.
    """

    def __init__(
        self,
        radio_maps: Union[RadioMapSet, np.ndarray, PathLike, None] = None,
        aggregation: str = "max",
        utility_c: float = 1e8,
        *,
        locations: Union[np.ndarray, PathLike, None] = None,
        density: Optional[np.ndarray] = None,
        basis_folder: Union[PathLike, None] = None,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        agg = str(aggregation).lower()
        if agg not in {"max", "sum"}:
            raise ValueError(f"aggregation must be 'max' or 'sum', got '{aggregation}'")
        self.aggregation = agg
        self.utility_c = float(utility_c)
        self.chunk_bytes = int(chunk_bytes)

        if radio_maps is None:
            radio_maps = basis_folder
        if radio_maps is None:
            raise ValueError(
                "Provide radio maps: an array, a file path, a basis-function "
                "folder, or a RadioMapSet."
            )

        self.basis_folder: Optional[Path] = None
        self.sionna_dir: Optional[Path] = None

        if isinstance(radio_maps, RadioMapSet):
            self.maps = radio_maps
        elif isinstance(radio_maps, (str, Path)):
            folder = Path(radio_maps)
            self.maps = RadioMapSet.from_files(folder, locations)
            if folder.is_dir():
                self.basis_folder = folder
                self.sionna_dir = folder / "Sionna" if (folder / "Sionna").is_dir() else folder
        else:
            coords = coerce_locations(locations)
            if coords is None:
                raise ValueError(
                    "'locations' is required when radio maps are given as an array."
                )
            self.maps = RadioMapSet(maps=np.asarray(radio_maps), locations=coords)

        self.coords = self.maps.locations
        self.n_candidates = self.maps.n_maps
        self.density = self._prepare_density(density)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_density(self, density: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Validate a spatial density and flatten it; None means uniform."""
        if density is None:
            return None
        density = np.asarray(density, dtype=float).ravel()
        if density.size != self.maps.n_cells:
            raise ValueError(
                f"density has {density.size} cells but the radio maps have "
                f"{self.maps.n_cells} (shape {self.maps.map_shape})."
            )
        if np.any(density < 0) or not np.all(np.isfinite(density)):
            raise ValueError("density must be finite and non-negative.")
        total = density.sum()
        if total <= 0:
            raise ValueError("density must have positive total mass.")
        return density / total

    def _resolve_density(self, density: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Per-call density override, falling back to the instance default."""
        return self.density if density is None else self._prepare_density(density)

    def _load_field(self, idx: int) -> np.ndarray:
        """Return the radio map of candidate *idx* in its original shape."""
        return self.maps.map_at(idx)

    def _aggregate_flat(self, u: np.ndarray) -> np.ndarray:
        """Aggregated field P(y, T) as a flat vector of ``n_cells`` values."""
        u = np.asarray(u, dtype=float).ravel()
        if u.size != self.n_candidates:
            raise ValueError(
                f"u has length {u.size}, expected {self.n_candidates} "
                f"(one entry per candidate)."
            )
        result = np.zeros(self.maps.n_cells, dtype=float)
        active = np.flatnonzero(u > 0)
        for idx in active:
            field = self.maps.block(idx, idx + 1)[0]
            if self.aggregation == "max":
                np.maximum(result, field, out=result)
            else:
                result += u[idx] * field
        return result

    def _expected_utility(
        self, field: np.ndarray, density: Optional[np.ndarray]
    ) -> float:
        """S = E[W(field)] under *density* (uniform when None)."""
        utility = w_bar(field, self.utility_c)
        if density is None:
            return float(utility.mean())
        return float(utility @ density)

    def _combine_inplace(self, block: np.ndarray, base: np.ndarray) -> np.ndarray:
        """Aggregate each row of *block* with the current field *base*."""
        if self.aggregation == "max":
            return np.maximum(block, base, out=block)
        block += base
        return block

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(self, u: np.ndarray) -> np.ndarray:
        """Compute the aggregated field P(y, T) for transmitter weights *u*.

        Parameters
        ----------
        u : np.ndarray, shape (n_candidates,)
            Indicator / weight vector.  Any entry > 0 activates the
            corresponding candidate.  Under ``"sum"`` aggregation the entry
            also scales that transmitter's contribution.

        Returns
        -------
        np.ndarray
            Radio map of the aggregated received power / rate, in the same
            shape as the input radio maps.
        """
        return self._aggregate_flat(u).reshape(self.maps.map_shape)

    def network_quality(
        self, u: np.ndarray, density: Optional[np.ndarray] = None
    ) -> float:
        """Evaluate S(T) = E_X[W(P(X, T))].

        This is the aggregated network quality functional defined in
        Theorem II.3 of the paper.

        Parameters
        ----------
        u : np.ndarray, shape (n_candidates,)
            Transmitter weight vector.
        density : np.ndarray or None
            Spatial priority density f(y).  If *None*, the optimiser's own
            density is used, which is uniform unless one was supplied to the
            constructor.

        Returns
        -------
        float
            Scalar value of S(T) in [0, 1).
        """
        return self._expected_utility(
            self._aggregate_flat(u), self._resolve_density(density)
        )

    def marginal_gains(
        self,
        u: np.ndarray,
        density: Optional[np.ndarray] = None,
        progress: bool = False,
    ) -> np.ndarray:
        """Compute G(x|T) = S(T u {x}) - S(T) for every candidate x.

        The current field P(y, T) is aggregated once and then combined with
        each candidate in turn, so one call costs a single pass over the radio
        maps regardless of how many transmitters are already placed.

        Parameters
        ----------
        u : np.ndarray, shape (n_candidates,)
            Current transmitter weight vector representing T.
        density : np.ndarray or None
            Spatial priority density.
        progress : bool
            Show a progress bar.

        Returns
        -------
        np.ndarray, shape (n_candidates,)
            Marginal gain for each candidate location.
        """
        density = self._resolve_density(density)
        base = self._aggregate_flat(u)
        s_base = self._expected_utility(base, density)
        return self._gains_from_field(base, s_base, density, progress=progress)

    def _gains_from_field(
        self,
        base: np.ndarray,
        s_base: float,
        density: Optional[np.ndarray],
        progress: bool = False,
    ) -> np.ndarray:
        """Marginal gains against an already-aggregated field *base*."""
        gains = np.empty(self.n_candidates, dtype=float)
        # A progress bar only earns its keep on problems big enough to take
        # noticeable time; on small ones it just clutters the output.
        show_bar = progress and self.n_candidates * self.maps.n_cells > _PROGRESS_MIN_WORK
        bar = tqdm(
            total=self.n_candidates,
            desc="Scoring candidates",
            disable=not show_bar,
            leave=False,
        )
        with bar:
            for start, stop, block in self.maps.blocks(self.chunk_bytes):
                combined = self._combine_inplace(block, base)
                # W(x) = x / (x + c), evaluated in place to keep the working
                # set at two blocks rather than four.
                denominator = combined + self.utility_c
                utility = np.divide(combined, denominator, out=combined)
                if density is None:
                    gains[start:stop] = utility.mean(axis=1) - s_base
                else:
                    gains[start:stop] = utility @ density - s_base
                bar.update(stop - start)
        return gains

    def snap_to_candidates(self, towers: np.ndarray) -> np.ndarray:
        """Map arbitrary tower coordinates to their nearest candidate index.

        Parameters
        ----------
        towers : np.ndarray, shape (m, 3) or (m, 2)
            Transmitter coordinates in the same frame as the candidates.

        Returns
        -------
        np.ndarray, shape (m,), dtype int
        """
        towers = _normalise_locations(np.asarray(towers, dtype=float), "fixed_towers")
        # A KD-tree keeps this linear in memory; the pairwise distance matrix
        # would be (m x n_candidates x 3), which does not scale.
        _, indices = cKDTree(self.coords).query(towers)
        return np.atleast_1d(indices).astype(int)

    def describe(self) -> str:
        """Summary of the loaded problem, for logs and sanity checks."""
        return (
            f"{self.maps.describe()}\n"
            f"Aggregation '{self.aggregation}', utility c = {self.utility_c:g}, "
            f"density {'uniform' if self.density is None else 'user-supplied'}."
        )


# ---------------------------------------------------------------------------
# Greedy selection
# ---------------------------------------------------------------------------

def greedy_select(
    optimizer: TowerOptimizer,
    n_towers: int,
    fixed_towers: Optional[np.ndarray] = None,
    density: Optional[np.ndarray] = None,
    allow_repeats: Optional[bool] = None,
    progress: bool = True,
    verbose: bool = True,
    on_iteration: Optional[Callable[[int, np.ndarray, int, np.ndarray], None]] = None,
) -> PlacementResult:
    """Run the IA-SPA greedy loop and return the selected transmitter sites.

    At each iteration the candidate with the highest marginal gain is added to
    the transmitter set T.  When *fixed_towers* is given, those locations are
    snapped to their nearest candidates and used as the warm start T_fixed
    (Section II-C of the paper).

    Parameters
    ----------
    optimizer : TowerOptimizer
        Configured optimiser instance.
    n_towers : int
        Number of new transmitters to place.
    fixed_towers : np.ndarray, shape (m, 3), optional
        Pre-existing transmitter coordinates to warm-start from.
    density : np.ndarray, optional
        Spatial priority density; overrides the optimiser's own.
    allow_repeats : bool, optional
        Whether a candidate site may be selected more than once.  Defaults to
        ``False`` for ``"max"`` aggregation, where re-selecting a site is
        provably a no-op, and to ``True`` for ``"sum"``, where co-locating
        transmitters does add power.
    progress : bool
        Show a per-iteration progress bar.
    verbose : bool
        Print the selection made at each iteration.
    on_iteration : callable, optional
        Called as ``on_iteration(iteration, gains, best_idx, u)`` after each
        selection, e.g. to persist intermediate state.

    Returns
    -------
    PlacementResult

    Notes
    -----
    The loop stops early -- with a warning -- if no remaining candidate has a
    positive marginal gain, since placing a transmitter that adds nothing to
    S(T) is not meaningful.
    """
    n_towers = int(n_towers)
    if n_towers < 0:
        raise ValueError(f"n_towers must be non-negative, got {n_towers}.")
    if allow_repeats is None:
        allow_repeats = optimizer.aggregation == "sum"

    density = optimizer._resolve_density(density)
    u = np.zeros(optimizer.n_candidates, dtype=float)

    # Scoring goes through the public marginal_gains() so that subclasses which
    # override it -- to forbid placement inside an exclusion zone, say -- still
    # take effect.  Such an override may predate the `progress` parameter, so
    # only pass it when the signature accepts it.
    scorer = optimizer.marginal_gains
    accepts_progress = "progress" in inspect.signature(scorer).parameters
    score_kwargs = {"progress": progress} if accepts_progress else {}

    # --- warm start ---------------------------------------------------
    fixed_indices = np.empty(0, dtype=int)
    if fixed_towers is not None and len(fixed_towers) > 0:
        fixed_indices = optimizer.snap_to_candidates(np.asarray(fixed_towers))
        for idx in fixed_indices:
            u[idx] += 1
        if verbose:
            unique = len(np.unique(fixed_indices))
            note = "" if unique == len(fixed_indices) else f" ({unique} distinct sites)"
            print(f"Warm start: {len(fixed_indices)} pre-placed transmitter(s){note}.")

    field = optimizer._aggregate_flat(u)
    s_current = optimizer._expected_utility(field, density)

    indices: list[int] = []
    positions: list[np.ndarray] = []
    step_gains: list[float] = []
    quality = [s_current]

    for iteration in range(n_towers):
        gains = np.asarray(scorer(u, density, **score_kwargs), dtype=float)
        if gains.shape != (optimizer.n_candidates,):
            raise ValueError(
                f"marginal_gains returned shape {gains.shape}, expected "
                f"({optimizer.n_candidates},)."
            )

        selectable = gains.copy()
        if not allow_repeats and len(indices) + len(fixed_indices):
            used = np.concatenate([fixed_indices, np.array(indices, dtype=int)])
            selectable[used] = -np.inf

        best_idx = int(np.argmax(selectable))
        best_gain = float(gains[best_idx])

        if not np.isfinite(selectable[best_idx]):
            warnings.warn(
                f"Stopping after {iteration} of {n_towers} placements: every "
                f"candidate site is already in use and repeats are disabled.",
                stacklevel=2,
            )
            break
        if best_gain <= 0.0:
            warnings.warn(
                f"Stopping after {iteration} of {n_towers} placements: the best "
                f"remaining marginal gain is {best_gain:.3g}, so no further "
                f"transmitter improves S(T).",
                stacklevel=2,
            )
            break

        u[best_idx] += 1
        best_field = optimizer.maps.block(best_idx, best_idx + 1)[0]
        if optimizer.aggregation == "max":
            np.maximum(field, best_field, out=field)
        else:
            field += best_field
        s_current = optimizer._expected_utility(field, density)

        indices.append(best_idx)
        positions.append(optimizer.coords[best_idx])
        step_gains.append(best_gain)
        quality.append(s_current)

        if verbose:
            x, y, z = optimizer.coords[best_idx]
            print(
                f"[{iteration + 1:>3}/{n_towers}] candidate #{best_idx}: "
                f"({x:.2f}, {y:.2f}, {z:.2f})  |  gain = {best_gain:.6f}  |  "
                f"S(T) = {s_current:.6f}"
            )

        if on_iteration is not None:
            on_iteration(iteration, gains, best_idx, u)

    return PlacementResult(
        positions=np.asarray(positions, dtype=float).reshape(-1, 3),
        indices=np.asarray(indices, dtype=int),
        gains=np.asarray(step_gains, dtype=float),
        quality=np.asarray(quality, dtype=float),
        u=u,
        fixed_positions=optimizer.coords[fixed_indices].reshape(-1, 3),
        fixed_indices=fixed_indices,
        aggregation=optimizer.aggregation,
        utility_c=optimizer.utility_c,
        n_candidates=optimizer.n_candidates,
    )


# ---------------------------------------------------------------------------
# Greedy runner (writes the on-disk result layout)
# ---------------------------------------------------------------------------

def run_greedy(
    optimizer: TowerOptimizer,
    results_folder: PathLike,
    n_iter: int,
    fixed_towers: Optional[np.ndarray] = None,
    density: Optional[np.ndarray] = None,
    allow_repeats: Optional[bool] = None,
    progress: bool = True,
    verbose: bool = True,
) -> PlacementResult:
    """Run :func:`greedy_select` and persist the results to disk.

    Parameters
    ----------
    optimizer : TowerOptimizer
        Configured optimiser instance.
    results_folder : str or Path
        Output directory.  Will be created if it does not exist.  The run
        writes:

        * ``Locations.txt``          - accumulated (x, y, z) of selected towers.
        * ``towers.csv``             - the same, with candidate index and gain.
        * ``u.npy``                  - final weight vector over all candidates.
        * ``Gain_Function_<k>.npy``  - full gain vector at iteration *k*.
        * ``summary.json``           - settings, selections and S(T) trajectory.
    n_iter : int
        Number of new transmitters to place.
    fixed_towers, density, allow_repeats, progress, verbose :
        Forwarded to :func:`greedy_select`.

    Returns
    -------
    PlacementResult
    """
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    locations_path = results_folder / "Locations.txt"
    locations_path.write_text("")

    def _persist(iteration: int, gains: np.ndarray, best_idx: int, u: np.ndarray) -> None:
        x, y, z = optimizer.coords[best_idx]
        with open(locations_path, "a") as fh:
            fh.write(f"{x}, {y}, {z}\n")
        np.save(results_folder / "u.npy", u)
        np.save(results_folder / f"Gain_Function_{iteration}.npy", gains)

    result = greedy_select(
        optimizer,
        n_towers=n_iter,
        fixed_towers=fixed_towers,
        density=density,
        allow_repeats=allow_repeats,
        progress=progress,
        verbose=verbose,
        on_iteration=_persist,
    )

    np.save(results_folder / "u.npy", result.u)
    save_positions(
        results_folder / "towers.csv", result.positions, result.indices, result.gains
    )
    (results_folder / "summary.json").write_text(json.dumps(result.to_dict(), indent=2))

    if verbose:
        print(f"\nDone. Results saved to '{results_folder}'.")
    return result
