"""
scripts/run_optimizer.py
------------------------
Run the IA-SPA greedy optimiser on pre-computed basis functions.

Usage
-----
    # Standard run (max aggregation, 20 new towers):
    python -m scripts.run_optimizer \\
        --basis BasisFunctions/SF \\
        --results Results/SF_max \\
        --n-iter 20 \\
        --aggregation max

    # Warm-start from pre-placed towers (incremental deployment):
    python -m scripts.run_optimizer \\
        --basis BasisFunctions/FL \\
        --results Results/FL_warmstart \\
        --n-iter 20 \\
        --preplaced-towers TowerData/FL_Iliad.npy

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
        "--preplaced-towers",
        default=None,
        help="Path to a .npy (N×3) or whitespace-delimited .txt file of "
             "pre-placed tower XYZ positions to warm-start from. "
             "If omitted, no towers are pre-placed.",
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

    basis_folder = Path(args.basis)
    results_folder = Path(args.results)
    results_folder.mkdir(parents=True, exist_ok=True)

    scene_file = basis_folder / "scene.txt"
    if scene_file.exists():
        (results_folder / "scene.txt").write_text(scene_file.read_text())

    optimizer = TowerOptimizer(
        basis_folder=basis_folder,
        aggregation=args.aggregation,
        utility_c=args.utility_c,
    )

    fixed_towers = None
    if args.preplaced_towers is not None:
        p = Path(args.preplaced_towers)
        fixed_towers = np.load(p) if p.suffix == ".npy" else np.loadtxt(p)
        print(f"Using {len(fixed_towers)} pre-placed transmitter(s) as warm-start.")

    run_greedy(
        optimizer=optimizer,
        results_folder=results_folder,
        n_iter=args.n_iter,
        fixed_towers=fixed_towers,
    )


if __name__ == "__main__":
    main()
