"""
ia_spa/optimizer.py
-------------------
Core implementation of the Interference-Aware Submodular Placement Algorithm (IA-SPA).

This module provides the `TowerOptimizer` class, which encapsulates the aggregated
network quality functional S(T) and the greedy placement loop described in:

    Taus, Tsai, Andrews. "Optimal Transmitter Placement in Realistic Urban
    Environments." (2026).

The key algorithmic idea is to iteratively select the candidate transmitter location
that maximises the marginal gain G(x|T) = S(T ∪ {x}) − S(T), where S(T) is the
expectation of the concave utility W̄ applied to the aggregated path-gain field.

Usage
-----
    from ia_spa.optimizer import TowerOptimizer, run_greedy

    optimizer = TowerOptimizer("path/to/BasisFunctions", aggregation="max")
    run_greedy(optimizer, results_folder="Results/run1", n_iter=20)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Utility functional
# ---------------------------------------------------------------------------

def w_bar(x: np.ndarray, c: float = 1e8) -> np.ndarray:
    """Concave utility function W̄(x) = x / (x + c).

    This satisfies the assumptions of Theorem II.4 in the paper: W̄ is
    non-negative, non-decreasing, and concave, which guarantees submodularity
    of S(T) and the greedy approximation bound of Theorem III.2.

    Parameters
    ----------
    x : np.ndarray
        Array of aggregated path-gain (or rate) values.
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
# Main optimiser class
# ---------------------------------------------------------------------------

class TowerOptimizer:
    """Evaluates S(T) and the marginal gain G(x|T) over pre-computed ray-traced fields.

    The constructor reads candidate transmitter coordinates from the standard
    ``0_Coordinates.txt`` file written by the basis-function scripts, and loads
    individual radio-map slices on demand using memory-mapped NumPy arrays to
    keep peak RAM usage bounded.

    Parameters
    ----------
    basis_folder : str or Path
        Root folder produced by a ``basefunctions_*.py`` run.  Must contain a
        ``Sionna/`` sub-directory with ``0_Coordinates.txt`` and one ``.npy``
        file per candidate location.
    aggregation : {"max", "sum"}
        Aggregation rule for the multi-transmitter field P(y, T):

        * ``"max"``  – P_MAX(y, T) = max_{t∈T} P(y,t).  Corresponds to a
          user associating with the strongest BS; the realistic default.
        * ``"sum"``  – P_SUM(y, T) = Σ_{t∈T} P(y,t).  Idealistic upper bound
          that treats all signals as constructive.
    utility_c : float, optional
        Saturation constant for W̄.  See :func:`w_bar`.
    """

    def __init__(
        self,
        basis_folder: str | Path,
        aggregation: str = "max",
        utility_c: float = 1e8,
    ) -> None:
        self.basis_folder = Path(basis_folder)
        self.sionna_dir = self.basis_folder / "Sionna"

        agg = aggregation.lower()
        if agg not in {"max", "sum"}:
            raise ValueError(f"aggregation must be 'max' or 'sum', got '{aggregation}'")
        self.aggregation = agg
        self.utility_c = utility_c

        # Load candidate transmitter coordinates (x, y, z)
        coord_file = self.sionna_dir / "0_Coordinates.txt"
        if not coord_file.exists():
            raise FileNotFoundError(f"Coordinate file not found: {coord_file}")

        with open(coord_file) as f:
            self.coords = np.array(
                [[float(v) for v in line.strip().split(", ")[1:]] for line in f]
            )

        self.n_candidates = len(self.coords)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_field(self, idx: int) -> np.ndarray:
        """Memory-map the rate field for candidate *idx*."""
        path = self.sionna_dir / f"{idx}.npy"
        return np.load(path, mmap_mode="r")

    def aggregate(self, u: np.ndarray) -> np.ndarray:
        """Compute the aggregated field P(y, T) for transmitter weights *u*.

        Parameters
        ----------
        u : np.ndarray, shape (n_candidates,)
            Indicator / weight vector.  Any entry > 0 activates the
            corresponding candidate.

        Returns
        -------
        np.ndarray
            2-D radio map representing the aggregated received power / rate.
        """
        # Initialise from the first field to get correct shape
        result = np.zeros_like(self._load_field(0))

        for i, weight in enumerate(u):
            if weight <= 0:
                continue
            field = np.asarray(self._load_field(i))
            if self.aggregation == "max":
                result = np.maximum(result, field)
            else:  # sum
                result = result + weight * field

        return result

    # ------------------------------------------------------------------
    # Network quality functional S(T)
    # ------------------------------------------------------------------

    def network_quality(
        self, u: np.ndarray, density: Optional[np.ndarray] = None
    ) -> float:
        """Evaluate S(T) = E_X[W̄(P(X, T))].

        This is the aggregated network quality functional defined in
        Theorem II.3 of the paper.

        Parameters
        ----------
        u : np.ndarray, shape (n_candidates,)
            Transmitter weight vector.
        density : np.ndarray or None
            Spatial priority density f(y).  If *None*, a uniform distribution
            is used (equivalent to averaging over all map cells).

        Returns
        -------
        float
            Scalar value of S(T) ∈ [0, 1).
        """
        field = self.aggregate(u)
        if density is None:
            density = np.ones_like(field) / field.size
        return float(np.sum(w_bar(field, self.utility_c) * density))

    # ------------------------------------------------------------------
    # Marginal gain G(x|T)
    # ------------------------------------------------------------------

    def marginal_gains(
        self, u: np.ndarray, density: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute G(x|T) for every candidate x ∈ X.

        G(x|T) = S(T ∪ {x}) − S(T).

        Parameters
        ----------
        u : np.ndarray, shape (n_candidates,)
            Current transmitter weight vector representing T.
        density : np.ndarray or None
            Spatial priority density.

        Returns
        -------
        np.ndarray, shape (n_candidates,)
            Marginal gain for each candidate location.
        """
        gains = np.zeros(self.n_candidates)
        s_base = self.network_quality(u, density)

        for i in tqdm(range(self.n_candidates), desc="Computing marginal gains"):
            v = u.copy()
            v[i] += 1
            gains[i] = self.network_quality(v, density) - s_base

        return gains


# ---------------------------------------------------------------------------
# Greedy runner
# ---------------------------------------------------------------------------

def run_greedy(
    optimizer: TowerOptimizer,
    results_folder: str | Path,
    n_iter: int,
    fixed_towers: Optional[np.ndarray] = None,
    density: Optional[np.ndarray] = None,
) -> None:
    """Run the IA-SPA greedy loop and persist results to disk.

    At each iteration the candidate with the highest marginal gain is selected
    and appended to the growing transmitter set T.  When *fixed_towers* is
    provided, those locations are snapped to the nearest candidates and used as
    the warm-start T_fixed (Section II-C of the paper).

    Parameters
    ----------
    optimizer : TowerOptimizer
        Configured optimiser instance.
    results_folder : str or Path
        Output directory.  Will be created if it does not exist.  Each
        iteration writes:

        * ``Locations.txt``   – accumulated (x, y, z) of selected towers.
        * ``u.npy``           – current weight vector.
        * ``Gain_Function_<k>.npy`` – full gain vector at iteration *k*.

    n_iter : int
        Number of new transmitters to place.
    fixed_towers : np.ndarray, shape (m, 3) or None
        Pre-existing transmitter coordinates to warm-start from (T_fixed).
    density : np.ndarray or None
        Spatial priority density passed through to :meth:`TowerOptimizer.network_quality`.
    """
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    u = np.zeros(optimizer.n_candidates)

    # Warm-start: snap fixed towers to nearest candidates
    if fixed_towers is not None and len(fixed_towers) > 0:
        print(f"Warm-starting from {len(fixed_towers)} fixed transmitter(s).")
        for tower in fixed_towers:
            distances = np.linalg.norm(optimizer.coords - tower, axis=1)
            nearest_idx = int(np.argmin(distances))
            u[nearest_idx] += 1

    # Clear / initialise the locations log
    locations_path = results_folder / "Locations.txt"
    locations_path.write_text("")

    for iteration in range(n_iter):
        print(f"\n--- Iteration {iteration + 1} / {n_iter} ---")

        gains = optimizer.marginal_gains(u, density)
        best_idx = int(np.argmax(gains))
        u[best_idx] += 1

        x, y, z = optimizer.coords[best_idx]
        with open(locations_path, "a") as f:
            f.write(f"{x}, {y}, {z}\n")

        np.save(results_folder / "u.npy", u)
        np.save(results_folder / f"Gain_Function_{iteration}.npy", gains)

        print(
            f"  Selected candidate #{best_idx}: ({x:.2f}, {y:.2f}, {z:.2f})"
            f"  |  gain = {gains[best_idx]:.6f}"
        )

    print(f"\nDone. Results saved to '{results_folder}'.")
