"""
scripts/run_optimizer.py
------------------------
Run the IA-SPA greedy optimiser on pre-computed basis functions.

Usage
-----
    # Standard run (max aggregation, 20 new towers):
    python scripts/run_optimizer.py \\
        --basis BasisFunctions/SF \\
        --results Results/SF_max \\
        --n-iter 20 \\
        --aggregation max

    # Warm-start from existing Iliad towers (Florence fixed-deployment scenario):
    python scripts/run_optimizer.py \\
        --basis BasisFunctions/FL \\
        --results Results/FL_warmstart \\
        --n-iter 20 \\
        --fixed-towers TowerData/FL_Iliad.npy \\
        --fixed-tower-indices 1 5 8

The script writes to *results*:
    Locations.txt        — (x, y, z) of each selected transmitter, one per line
    u.npy                — final weight vector over all candidates
    Gain_Function_k.npy  — gain vector G(·|T_k) at each iteration k
"""

import argparse
from pathlib import Path

import numpy as np

from ia_spa.optimizer import TowerOptimizer, run_greedy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the IA-SPA greedy optimiser.")
    p.add_argument("--basis", required=True, help="Basis-function folder")
    p.add_argument("--results", required=True, help="Output folder for results")
    p.add_argument("--n-iter", type=int, default=20, help="Number of towers to place")
    p.add_argument(
        "--aggregation",
        choices=["max", "sum"],
        default="max",
        help="Aggregation rule: 'max' (P_MAX) or 'sum' (P_SUM)",
    )
    p.add_argument(
        "--fixed-towers",
        default=None,
        help="Path to .npy file of fixed (pre-existing) tower positions (N×3)",
    )
    p.add_argument(
        "--fixed-tower-indices",
        nargs="+",
        type=int,
        default=None,
        help="Row indices into --fixed-towers to use (default: all rows)",
    )
    p.add_argument(
        "--utility-c",
        type=float,
        default=1e8,
        help="Saturation constant c in W̄(x) = x/(x+c)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    optimizer = TowerOptimizer(
        basis_folder=args.basis,
        aggregation=args.aggregation,
        utility_c=args.utility_c,
    )

    fixed_towers = None
    if args.fixed_towers is not None:
        towers = np.load(args.fixed_towers)
        if args.fixed_tower_indices is not None:
            towers = towers[args.fixed_tower_indices, :]
        fixed_towers = towers
        print(f"Using {len(fixed_towers)} fixed transmitter(s) as warm-start.")

    run_greedy(
        optimizer=optimizer,
        results_folder=args.results,
        n_iter=args.n_iter,
        fixed_towers=fixed_towers,
    )


if __name__ == "__main__":
    main()
