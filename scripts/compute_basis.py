"""
scripts/compute_basis.py
------------------------
Pre-compute per-candidate ray-traced rate maps for any Sionna scene.

Usage
-----
    # Built-in San Francisco scene, uniform 3-D grid:
    python -m scripts.compute_basis \\
        --scene san_francisco \\
        --output data/BasisFunctions/SF \\
        --dx 20 --dy 20 --dz 20

    # Built-in Florence scene, same grid:
    python -m scripts.compute_basis \\
        --scene florence \\
        --output data/BasisFunctions/FL \\
        --dx 15 --dy 15 --dz 10

    # Custom scene from an XML file:
    python -m scripts.compute_basis \\
        --scene path/to/my_scene.xml \\
        --output data/BasisFunctions/MyScene \\
        --dx 20 --dy 20 --dz 20

    # User-supplied candidate positions (N x 3 .npy or whitespace .txt):
    python -m scripts.compute_basis \\
        --scene san_francisco \\
        --output data/BasisFunctions/SF_custom \\
        --user-positions path/to/candidates.npy

    # Resume an interrupted run:
    python -m scripts.compute_basis \\
        --scene san_francisco \\
        --output data/BasisFunctions/SF \\
        --resume

All physical parameters are read from ``config/config.yaml`` and can be
overridden individually with CLI flags.

Candidate Modes
---------------
If ``--user-positions`` is given, those coordinates are used directly.
Otherwise a uniform (dx, dy, dz) grid is placed over the full scene
bounding box.  Candidates that fall inside geometry are kept -- they will
produce near-zero radio maps and will never be selected by the optimizer.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from sionna.rt import load_scene, scene as sionna_scenes

from ia_spa.basis_functions import (
    candidate_locations_plot,
    compute_basis_functions,
    compute_basis_functions_resume,
    get_candidate_locations,
)

# Short aliases for the built-in Sionna scenes
_BUILTIN_SCENES = {
    "san_francisco": "san_francisco",
    "sf":            "san_francisco",
    "florence":      "florence",
    "fl":            "florence",
    "munich":        "munich",
    "etoile":        "etoile",
}


def _load_scene(scene_arg: str, merge_shapes: bool):
    """Load a built-in scene by name or a custom scene from an XML path."""
    key = scene_arg.lower()
    if key in _BUILTIN_SCENES:
        scene_id = getattr(sionna_scenes, _BUILTIN_SCENES[key])
        return load_scene(scene_id, merge_shapes=merge_shapes)
    path = Path(scene_arg)
    if not path.exists():
        sys.exit(
            f"ERROR: '{scene_arg}' is not a recognised built-in scene name "
            f"and the file does not exist.\n"
            f"Built-in names: {list(_BUILTIN_SCENES.keys())}"
        )
    return load_scene(str(path), merge_shapes=merge_shapes)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute IA-SPA basis functions for any Sionna scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scene", required=True,
        help="Built-in scene name (e.g. 'san_francisco', 'florence') "
             "or path to a Sionna XML scene file.",
    )
    p.add_argument(
        "--output", required=True,
        help="Output folder for basis functions.",
    )
    p.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config.yaml.",
    )

    # Candidate selection
    p.add_argument(
        "--user-positions", default=None,
        help="Path to a .npy (N x 3) or whitespace-delimited .txt file of "
             "candidate XYZ positions.  If omitted, a uniform grid is used.",
    )
    p.add_argument("--dx", type=float, default=None, help="Grid X step size in metres.")
    p.add_argument("--dy", type=float, default=None, help="Grid Y step size in metres.")
    p.add_argument("--dz", type=float, default=None, help="Grid Z step size in metres.")

    # Run control
    p.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted run from the last completed candidate.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    output_folder  = Path(args.output)
    power_dbm      = float(cfg["transmitter"]["power_dbm"])
    frequency_hz   = float(cfg["scene"]["frequency_hz"])
    bandwidth_hz   = float(cfg["link"]["bandwidth_hz"])
    snr_gap_gamma  = float(cfg["link"]["snr_gap_gamma"])
    max_depth      = int(cfg["radio_map"]["max_depth"])
    samples_per_tx = int(cfg["radio_map"]["samples_per_tx"])

    cand_cfg = cfg.get("candidates", {})
    dx = args.dx if args.dx is not None else float(cand_cfg.get("dx_m", 20.0))
    dy = args.dy if args.dy is not None else float(cand_cfg.get("dy_m", 20.0))
    dz = args.dz if args.dz is not None else float(cand_cfg.get("dz_m", 20.0))

    print(
        f"Config: power={power_dbm} dBm, freq={frequency_hz/1e9:.3f} GHz, "
        f"BW={bandwidth_hz/1e6:.1f} MHz, samples_per_tx={samples_per_tx:,}, "
        f"max_depth={max_depth}"
    )

    sionna_dir = output_folder / "Sionna"
    coord_file = sionna_dir / "0_Coordinates.txt"

    # ------------------------------------------------------------------
    # Step 1: Determine candidate locations
    # ------------------------------------------------------------------
    if args.resume:
        # Validate that a previous run exists to resume from
        if not coord_file.exists():
            sys.exit(
                f"ERROR: --resume specified but no existing run found at '{output_folder}'.\n"
                f"Run without --resume to start a fresh computation."
            )
        print(f"Resuming existing run in '{output_folder}'.")

    else:
        # Fresh run: delete any existing output and recompute from scratch
        if output_folder.exists():
            ans = input(
                f"Output folder '{output_folder}' already exists.\n"
                f"Delete it and recompute from scratch? [y/N] "
            )
            if ans.strip().lower() != "y":
                sys.exit("Aborted.")
            shutil.rmtree(output_folder)
            print(f"Deleted '{output_folder}'.")

        # Compute candidate positions
        scene_for_bbox = _load_scene(args.scene, merge_shapes=False)

        if args.user_positions is not None:
            up = Path(args.user_positions)
            user_pos = np.load(up) if up.suffix == ".npy" else np.loadtxt(up)
            candidates = get_candidate_locations(
                scene_for_bbox, mode="user", user_positions=user_pos,
            )
        else:
            candidates = get_candidate_locations(
                scene_for_bbox, mode="grid", dx=dx, dy=dy, dz=dz,
            )

        output_folder.mkdir(parents=True, exist_ok=True)
        (output_folder / "scene.txt").write_text(args.scene)
        candidate_locations_plot(candidates, output_folder / "candidate_locations.html")

    # ------------------------------------------------------------------
    # Step 2: Compute (or resume) basis functions
    # ------------------------------------------------------------------
    scene_rt = _load_scene(args.scene, merge_shapes=True)

    params = dict(
        power_dbm=power_dbm,
        bandwidth_hz=bandwidth_hz,
        snr_gap_gamma=snr_gap_gamma,
        max_depth=max_depth,
        samples_per_tx=samples_per_tx,
        frequency_hz=frequency_hz,
    )

    if args.resume:
        compute_basis_functions_resume(output_folder, scene_rt, **params)
    else:
        compute_basis_functions(scene_rt, candidates, output_folder, **params)

    print("Done.")


if __name__ == "__main__":
    main()
