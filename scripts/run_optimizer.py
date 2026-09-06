"""
scripts/run_optimizer.py
------------------------
Run the IA-SPA greedy optimiser on pre-computed radio maps.

Usage
-----
    # From a stacked radio-map file plus its candidate locations:
    python -m scripts.run_optimizer \\
        --radio-maps radio_maps.npy \\
        --locations  locations.csv \\
        --results    Results/MyRun \\
        --n-iter 20

    # From a basis-function folder written by scripts/compute_basis.py:
    python -m scripts.run_optimizer \\
        --basis   data/BasisFunctions/SF \\
        --results data/Results/SF_max \\
        --n-iter 20 \\
        --aggregation max

    # Warm-start from pre-placed towers (incremental deployment):
    python -m scripts.run_optimizer \\
        --basis   data/BasisFunctions/FL \\
        --results data/Results/FL_warmstart \\
        --n-iter 20 \\
        --preplaced-towers data/TowerData/FL_Iliad.npy

The script writes to *results*:
    Locations.txt        - (x, y, z) of each selected transmitter, one per line
    towers.csv           - the same, with candidate index and marginal gain
    u.npy                - final weight vector over all candidates
    Gain_Function_k.npy  - gain vector G(.|T_k) at each iteration k
    summary.json         - settings, selections and the S(T) trajectory

For the shortest path from radio maps to tower locations, use the ``ia-spa``
command instead; it shares the same optimiser but skips the on-disk run layout.
"""

import argparse
import sys
from pathlib import Path

from ia_spa.data import load_locations
from ia_spa.optimizer import TowerOptimizer, run_greedy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the IA-SPA greedy optimiser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--basis",
        help="Basis-function folder written by scripts/compute_basis.py "
             "(its Sionna/0_Coordinates.txt supplies the candidate locations).",
    )
    source.add_argument(
        "--radio-maps",
        help="Stacked radio maps: .npy (N, H, W), .npz, or a text table. "
             "Requires --locations.",
    )
    p.add_argument(
        "--locations",
        default=None,
        help="Candidate site coordinates (N, 3), in the same order as the "
             "radio maps. Required with --radio-maps.",
    )
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
        help="Path to a .npy (N x 3) or text file of pre-placed tower XYZ "
             "positions to warm-start from. If omitted, no towers are pre-placed.",
    )
    p.add_argument(
        "--utility-c",
        type=float,
        default=1e8,
        help="Saturation constant c in W(x) = x/(x+c)",
    )
    args = p.parse_args()
    if args.radio_maps and args.locations is None:
        p.error("--locations is required when --radio-maps is used.")
    return args


def main() -> int:
    args = parse_args()

    results_folder = Path(args.results)
    results_folder.mkdir(parents=True, exist_ok=True)

    source = args.basis or args.radio_maps
    if args.basis:
        # Carry the scene name forward so scripts/evaluate.py can find it.
        scene_file = Path(args.basis) / "scene.txt"
        if scene_file.exists():
            (results_folder / "scene.txt").write_text(scene_file.read_text())

    try:
        optimizer = TowerOptimizer(
            source,
            aggregation=args.aggregation,
            utility_c=args.utility_c,
            locations=args.locations,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(optimizer.describe())

    fixed_towers = None
    if args.preplaced_towers is not None:
        fixed_towers = load_locations(args.preplaced_towers)

    run_greedy(
        optimizer=optimizer,
        results_folder=results_folder,
        n_iter=args.n_iter,
        fixed_towers=fixed_towers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
