"""
scripts/evaluate_sf.py
-----------------------
Evaluate IA-SPA results against reference AT&T / T-Mobile deployments
on the San Francisco scene and save SINR / rate / interference maps.

Usage
-----
    python scripts/evaluate_sf.py --results Results/SF_max [--config config.yaml]

For each reference carrier and for the greedy solution (truncated to the same
number of towers as each carrier), this script calls
:func:`ia_spa.metrics.evaluate_and_save` and stores ``.npy`` metric arrays in
``<results>_Processed/``.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from sionna.rt import PlanarArray, load_scene, scene as sionna_scenes

from ia_spa.basis_functions import apply_tower_height, build_terrain_tree
from ia_spa.metrics import evaluate_and_save, load_greedy_positions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="Greedy results folder")
    p.add_argument("--config", default="config.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_folder = Path(args.results)
    save_dir = results_folder.parent / (results_folder.name + "_Processed")

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    tower_height = float(cfg["transmitter"]["tower_height_m"])
    power_dbm = float(cfg["transmitter"]["power_dbm"])
    frequency_hz = float(cfg["scene"]["frequency_hz"])
    heights = cfg["receiver"]["heights_m"]
    bandwidth_hz = float(cfg["link"]["bandwidth_hz"])
    snr_gap_gamma = float(cfg["link"]["snr_gap_gamma"])

    # ------------------------------------------------------------------
    # Load scene
    # ------------------------------------------------------------------
    sf_scene = load_scene(sionna_scenes.san_francisco, merge_shapes=False)
    sf_scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    sf_scene.frequency = frequency_hz

    terrain_tree, terrain_verts = build_terrain_tree(sf_scene, "Terrain")

    # ------------------------------------------------------------------
    # Reference deployments
    # ------------------------------------------------------------------
    tower_data = Path(cfg["paths"]["tower_data"])
    refs = {
        "ATT_ref": np.load(tower_data / "SF_ATT.npy"),
        "TMobile_ref": np.load(tower_data / "SF_TMobile.npy"),
        "ATT_TMobile_ref": np.load(tower_data / "SF_ATT_TMobile.npy"),
    }

    if tower_height:
        refs = {k: apply_tower_height(v, terrain_tree, terrain_verts, tower_height)
                for k, v in refs.items()}

    for name, pos in refs.items():
        evaluate_and_save(sf_scene, name, pos, save_dir, heights, power_dbm, bandwidth_hz, snr_gap_gamma)

    # ------------------------------------------------------------------
    # Greedy solution (evaluated at each carrier's tower count)
    # ------------------------------------------------------------------
    greedy_pos = load_greedy_positions(results_folder)
    if tower_height:
        greedy_pos = apply_tower_height(greedy_pos, terrain_tree, terrain_verts, tower_height)

    for ref_name, ref_pos in refs.items():
        carrier = ref_name.replace("_ref", "")
        n = ref_pos.shape[0]
        evaluate_and_save(
            sf_scene, f"{carrier}_Greedy",
            greedy_pos[:n], save_dir,
            heights, power_dbm, bandwidth_hz, snr_gap_gamma,
        )

    print(f"All results saved to '{save_dir}'.")


if __name__ == "__main__":
    main()
