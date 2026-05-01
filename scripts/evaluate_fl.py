"""
scripts/evaluate_fl.py
-----------------------
Evaluate IA-SPA results against reference Iliad / TIM / Vodafone / WindTre
deployments on the Florence scene.

Usage
-----
    python scripts/evaluate_fl.py --results Results/FL_max [--config config.yaml]
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
    p.add_argument("--results", required=True)
    p.add_argument("--config", default="config/config.yaml")
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

    fl_scene = load_scene(sionna_scenes.florence, merge_shapes=False)
    fl_scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    fl_scene.frequency = frequency_hz

    terrain_tree, terrain_verts = build_terrain_tree(fl_scene, "ground")

    tower_data = Path(cfg["paths"]["tower_data"])
    refs = {
        "Iliad_ref": np.load(tower_data / "FL_Iliad.npy"),
        "TIM_ref": np.load(tower_data / "FL_TIM.npy"),
        "Vodafone_ref": np.load(tower_data / "FL_Vodafone.npy"),
        "WindTre_ref": np.load(tower_data / "FL_WindTre.npy"),
    }

    if tower_height:
        refs = {k: apply_tower_height(v, terrain_tree, terrain_verts, tower_height)
                for k, v in refs.items()}

    for name, pos in refs.items():
        evaluate_and_save(fl_scene, name, pos, save_dir, heights, power_dbm, bandwidth_hz, snr_gap_gamma)

    greedy_pos = load_greedy_positions(results_folder)
    if tower_height:
        greedy_pos = apply_tower_height(greedy_pos, terrain_tree, terrain_verts, tower_height)

    for ref_name, ref_pos in refs.items():
        carrier = ref_name.replace("_ref", "")
        n = ref_pos.shape[0]
        evaluate_and_save(
            fl_scene, f"{carrier}_Greedy",
            greedy_pos[:n], save_dir,
            heights, power_dbm, bandwidth_hz, snr_gap_gamma,
        )

    print(f"All results saved to '{save_dir}'.")


if __name__ == "__main__":
    main()
