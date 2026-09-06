"""
ia_spa/cli.py
-------------
Command-line interface for the IA-SPA placement algorithm.

Radio maps in, tower locations out::

    ia-spa --radio-maps radio_maps.npy --locations sites.csv --n-towers 15

The same command is available without installing the console script::

    python -m ia_spa --radio-maps radio_maps.npy --locations sites.csv -n 15

Inputs
------
``--radio-maps``
    One radio map per candidate transmitter site: an ``(N, H, W)`` ``.npy``
    stack, an ``.npz`` archive, a text table of N flattened maps, or a
    directory holding ``0.npy``, ``1.npy``, ...
``--locations``
    The ``(N, 3)`` (x, y, z) coordinates of those candidate sites, in the same
    order, as ``.npy`` / ``.csv`` / ``.txt``.

Output
------
A table of the chosen tower coordinates, written to ``--output``
(``towers.csv`` by default) and printed as a summary.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

from ia_spa.data import RadioMapSet
from ia_spa.placement import check_utility_scale, place_towers

EPILOG = """\
examples:
  # 15 towers from a stacked .npy of radio maps and a .csv of site coordinates
  ia-spa -m radio_maps.npy -l sites.csv -n 15 -o towers.csv

  # validate the inputs without running the optimiser
  ia-spa -m radio_maps.npy -l sites.csv --check-only

  # add 10 towers to an existing deployment, weighting by a population map
  ia-spa -m radio_maps.npy -l sites.csv -n 10 \\
         --fixed-towers existing_towers.csv --density population.npy

  # a directory of per-candidate maps written by scripts/compute_basis.py
  # (its 0_Coordinates.txt is picked up automatically)
  ia-spa -m data/BasisFunctions/SF -n 20 -o data/Results/SF_max
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ia-spa",
        description=(
            "IA-SPA: choose optimal transmitter sites from per-candidate radio "
            "maps. Input: a file of radio maps and a file of the matching "
            "candidate locations. Output: the locations of the chosen towers."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io_group = parser.add_argument_group("input / output")
    io_group.add_argument(
        "-m", "--radio-maps", required=True, metavar="PATH",
        help="Radio maps, one per candidate site: .npy (N, H, W) stack, .npz, "
             "text table, or a directory of {i}.npy files.",
    )
    io_group.add_argument(
        "-l", "--locations", default=None, metavar="PATH",
        help="Candidate site coordinates (N, 3) as .npy/.csv/.txt, in the same "
             "order as the radio maps. Optional only for a directory of maps "
             "that carries its own 0_Coordinates.txt.",
    )
    io_group.add_argument(
        "-o", "--output", default="towers.csv", metavar="PATH",
        help="Where to write the chosen tower locations: .csv, .txt or .npy, "
             "or a directory to receive towers.csv and summary.json. "
             "(default: towers.csv)",
    )

    algo = parser.add_argument_group("algorithm")
    algo.add_argument(
        "-n", "--n-towers", type=int, default=10, metavar="K",
        help="Number of transmitters to place. (default: 10)",
    )
    algo.add_argument(
        "-a", "--aggregation", choices=["max", "sum"], default="max",
        help="Multi-transmitter aggregation: 'max' (P_MAX, association with "
             "the strongest base station) or 'sum' (P_SUM, idealised upper "
             "bound). (default: max)",
    )
    algo.add_argument(
        "-c", "--utility-c", type=float, default=1e8, metavar="C",
        help="Saturation constant of W(x) = x/(x+c), in the units of the radio "
             "maps. (default: 1e8, i.e. 100 Mbps)",
    )
    algo.add_argument(
        "--fixed-towers", default=None, metavar="PATH",
        help="Coordinates of transmitters that already exist. They are held "
             "fixed, so the run returns the best additions to that deployment.",
    )
    algo.add_argument(
        "--density", default=None, metavar="PATH",
        help="Per-cell spatial priority weights (e.g. population density) with "
             "the same shape as one radio map. Uniform by default.",
    )
    algo.add_argument(
        "--allow-repeats", dest="allow_repeats", action="store_true", default=None,
        help="Allow the same candidate site to be selected more than once. "
             "(default under 'sum' aggregation)",
    )
    algo.add_argument(
        "--no-repeats", dest="allow_repeats", action="store_false",
        help="Forbid selecting the same candidate site twice. "
             "(default under 'max' aggregation)",
    )

    run = parser.add_argument_group("run control")
    run.add_argument(
        "--check-only", action="store_true",
        help="Load and validate the inputs, print a summary, and exit without "
             "running the optimiser.",
    )
    run.add_argument(
        "-q", "--quiet", action="store_true",
        help="Suppress per-iteration output and the progress bar.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``ia-spa`` console script.

    Returns
    -------
    int
        Process exit status: 0 on success, 2 when the inputs are unusable.
    """
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    try:
        maps = RadioMapSet.from_files(args.radio_maps, args.locations)
    except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if verbose:
        print(maps.describe())

    if args.check_only:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_utility_scale(maps, args.utility_c)
        for warning in caught:
            print(f"WARNING: {warning.message}")
        print(
            f"Inputs are valid: {maps.n_maps} candidate sites, "
            f"maps of shape {maps.map_shape}. Nothing was computed "
            f"(--check-only)."
        )
        return 0

    try:
        result = place_towers(
            maps,
            n_towers=args.n_towers,
            aggregation=args.aggregation,
            utility_c=args.utility_c,
            density=args.density,
            fixed_towers=args.fixed_towers,
            allow_repeats=args.allow_repeats,
            output=args.output,
            progress=verbose,
            verbose=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(result.summary())
    print(f"\nTower locations written to '{Path(args.output)}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
