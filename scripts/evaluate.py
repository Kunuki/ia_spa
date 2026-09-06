"""
scripts/evaluate.py
-------------------
Evaluate IA-SPA optimiser results for any Sionna scene.

Reads tower positions from a results folder produced by ``run_optimizer.py``,
places them in the scene, and saves SINR / rate / interference maps to
``<results>_Processed/``.  A summary (mean rate, edge rate, mean SINR) is
printed to stdout.

The scene is read automatically from ``<results>/scene.txt``, which is written
by ``run_optimizer.py``.  Pass ``--scene`` to override.

Usage
-----
    # Minimal — scene resolved from results/scene.txt:
    python -m scripts.evaluate --results data/Results/SF_max

    # Override scene:
    python -m scripts.evaluate \\
        --results data/Results/MyScene_max \\
        --scene path/to/my_scene.xml

    # Evaluate only the first N towers:
    python -m scripts.evaluate --results data/Results/SF_max --n-towers 10

    # Save a two-panel rate / SINR map PNG with tower locations overlaid:
    python -m scripts.evaluate --results data/Results/SF_max --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from ia_spa.metrics import evaluate_and_save, load_greedy_positions, plot_evaluation
from ia_spa.scenes import BUILTIN_SCENES, load_scene_arg


def _load_scene(scene_arg: str, frequency_hz: float):
    """Load the scene and give it the single isotropic antenna used throughout."""
    from sionna.rt import PlanarArray  # GPU / Sionna required

    return load_scene_arg(
        scene_arg,
        merge_shapes=False,
        frequency_hz=frequency_hz,
        tx_array=PlanarArray(
            num_rows=1, num_cols=1, pattern="iso", polarization="V"
        ),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate IA-SPA results for any Sionna scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results", required=True,
        help="Results folder produced by run_optimizer.py "
             "(must contain Locations.txt).",
    )
    p.add_argument(
        "--scene", default=None,
        help=f"Built-in scene name ({', '.join(sorted(BUILTIN_SCENES))}) or "
             f"path to a Sionna XML file. Defaults to the value stored in "
             f"<results>/scene.txt.",
    )
    p.add_argument(
        "--n-towers", type=int, default=None,
        help="Evaluate only the first N towers. Defaults to all.",
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Save a two-panel PNG showing the rate map and SINR map "
             "with placed towers overlaid (saved to <results>_Processed/).",
    )
    p.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config.yaml.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_folder = Path(args.results)
    save_dir = results_folder.parent / (results_folder.name + "_Processed")

    # ------------------------------------------------------------------
    # Resolve scene name
    # ------------------------------------------------------------------
    scene_arg = args.scene
    if scene_arg is None:
        scene_file = results_folder / "scene.txt"
        if not scene_file.exists():
            print(
                f"ERROR: No scene specified and '{scene_file}' not found.\n"
                f"Pass --scene to specify the scene explicitly.",
                file=sys.stderr,
            )
            return 2
        scene_arg = scene_file.read_text().strip()
        print(f"Scene resolved from results/scene.txt: '{scene_arg}'")

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    power_dbm     = float(cfg["transmitter"]["power_dbm"])
    frequency_hz  = float(cfg["scene"]["frequency_hz"])
    heights       = cfg["receiver"]["heights_m"]
    bandwidth_hz  = float(cfg["link"]["bandwidth_hz"])
    snr_gap_gamma = float(cfg["link"]["snr_gap_gamma"])
    cell_size_m   = float(cfg["radio_map"].get("cell_size_m", 1.0))

    # ------------------------------------------------------------------
    # Load scene
    # ------------------------------------------------------------------
    try:
        scene = _load_scene(scene_arg, frequency_hz)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # Load optimised tower positions
    # ------------------------------------------------------------------
    locations_file = results_folder / "Locations.txt"
    if not locations_file.exists():
        print(
            f"ERROR: '{locations_file}' not found. Run scripts/run_optimizer.py "
            f"first, or point --results at a finished run.",
            file=sys.stderr,
        )
        return 2

    positions = load_greedy_positions(results_folder)
    if args.n_towers is not None:
        positions = positions[: args.n_towers]
    if len(positions) == 0:
        print(f"ERROR: '{locations_file}' contains no tower positions.",
              file=sys.stderr)
        return 2

    print(f"Evaluating {len(positions)} tower(s) from '{results_folder}'.")

    # ------------------------------------------------------------------
    # Evaluate and save
    # ------------------------------------------------------------------
    run_name = results_folder.name
    evaluate_and_save(
        scene, run_name, positions, save_dir,
        heights, power_dbm, bandwidth_hz, snr_gap_gamma, cell_size_m,
    )

    if args.plot:
        rate = np.load(save_dir / f"{run_name}_rate.npy")
        sinr = np.load(save_dir / f"{run_name}_sinr.npy")
        bbox = scene._scene.bbox()
        scene_bbox = (bbox.min[0], bbox.min[1]), (bbox.max[0], bbox.max[1])
        plot_evaluation(
            rate, sinr, positions,
            save_path=save_dir / f"{run_name}_map.png",
            scene_bbox=scene_bbox,
        )

    print(f"Results saved to '{save_dir}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
