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
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from sionna.rt import PlanarArray, load_scene, scene as sionna_scenes

from ia_spa.metrics import evaluate_and_save, load_greedy_positions

_BUILTIN_SCENES = {
    "san_francisco": "san_francisco",
    "sf":            "san_francisco",
    "florence":      "florence",
    "fl":            "florence",
    "munich":        "munich",
    "etoile":        "etoile",
}


def _load_scene(scene_arg: str, frequency_hz: float):
    key = scene_arg.lower()
    if key in _BUILTIN_SCENES:
        scene_id = getattr(sionna_scenes, _BUILTIN_SCENES[key])
        s = load_scene(scene_id, merge_shapes=False)
    else:
        path = Path(scene_arg)
        if not path.exists():
            sys.exit(
                f"ERROR: '{scene_arg}' is not a recognised built-in scene name "
                f"and the file does not exist.\n"
                f"Built-in names: {list(_BUILTIN_SCENES.keys())}"
            )
        s = load_scene(str(path), merge_shapes=False)

    s.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    s.frequency = frequency_hz
    return s


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
        help="Built-in scene name or path to a Sionna XML file. "
             "Defaults to the value stored in <results>/scene.txt.",
    )
    p.add_argument(
        "--n-towers", type=int, default=None,
        help="Evaluate only the first N towers. Defaults to all.",
    )
    p.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config.yaml.",
    )
    return p.parse_args()


def main() -> None:
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
            sys.exit(
                "ERROR: No scene specified and results/scene.txt not found.\n"
                "Pass --scene to specify the scene explicitly."
            )
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

    # ------------------------------------------------------------------
    # Load scene
    # ------------------------------------------------------------------
    scene = _load_scene(scene_arg, frequency_hz)

    # ------------------------------------------------------------------
    # Load optimised tower positions
    # ------------------------------------------------------------------
    positions = load_greedy_positions(results_folder)
    if args.n_towers is not None:
        positions = positions[: args.n_towers]

    print(f"Evaluating {len(positions)} tower(s) from '{results_folder}'.")

    # ------------------------------------------------------------------
    # Evaluate and save
    # ------------------------------------------------------------------
    run_name = results_folder.name
    evaluate_and_save(
        scene, run_name, positions, save_dir,
        heights, power_dbm, bandwidth_hz, snr_gap_gamma,
    )

    print(f"Results saved to '{save_dir}'.")


if __name__ == "__main__":
    main()
