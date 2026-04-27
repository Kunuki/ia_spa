"""
ia_spa/basis_functions.py
-------------------------
Candidate transmitter-location selection and per-candidate radio-map
pre-computation via Sionna RT.

This module is intentionally simple: there is no SDF, no mesh processing,
and no obstacle filtering.  The rationale is that a transmitter placed inside
an obstacle will produce a near-zero radio map and will therefore never be
selected by the greedy algorithm -- no explicit filtering is needed.

Works with **any** Sionna scene:
  - Built-in scenes  (san_francisco, florence, munich, ...)
  - Custom scenes loaded from XML files
  - Any scene object already in memory

The scene bounding box is read directly from Sionna via
``scene._scene.bbox()``, so no scene-specific knowledge is required.

Candidate Location Modes
------------------------
Two modes are supported by :func:`get_candidate_locations`:

``"user"``
    The caller supplies an explicit (N, 3) array of candidate positions.
    These are used as-is with no modification or filtering.

``"grid"``
    A uniform 3-D grid with step sizes (dx, dy, dz) spanning the full
    scene bounding box.  Every grid node is kept as a candidate.
    Nodes that happen to fall inside an obstacle will produce weak radio
    maps and will not be selected by the optimizer.

Pipeline
--------
1. :func:`get_candidate_locations` -- produce the (N, 3) candidate array.
2. :func:`compute_basis_functions` -- ray-trace each candidate and save its
   rate map.  Resumable via :func:`compute_basis_functions_resume`.

Height utilities
----------------
:func:`build_terrain_tree` and :func:`apply_tower_height` are provided for
post-processing reference tower data (snapping real-world tower coordinates
to local terrain height).  They are not required by the optimisation loop.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Optional, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree
from tqdm import tqdm

from sionna.rt import (
    PlanarArray,
    RadioMapSolver,
    Transmitter,
)


# ---------------------------------------------------------------------------
# Candidate location selection
# ---------------------------------------------------------------------------

def get_candidate_locations(
    sionna_scene,
    mode: Literal["user", "grid"] = "grid",
    *,
    # ---- "user" mode ----
    user_positions: Optional[np.ndarray] = None,
    # ---- "grid" mode ----
    dx: float = 20.0,
    dy: float = 20.0,
    dz: float = 20.0,
) -> np.ndarray:
    """Return candidate transmitter locations for any Sionna scene.

    Parameters
    ----------
    sionna_scene :
        Any loaded Sionna scene object.  The bounding box is read via
        ``sionna_scene._scene.bbox()``.  The scene itself is not modified.
    mode : {"user", "grid"}
        Candidate-selection strategy:

        ``"user"``  Use *user_positions* exactly as supplied.
        ``"grid"``  Build a uniform 3-D grid over the scene bounding box.

    user_positions : np.ndarray, shape (N, 3), optional
        Required when ``mode="user"``.  World-space (x, y, z) coordinates
        in the same units as the Sionna scene (metres).

    dx, dy, dz : float
        Grid step sizes in metres, used when ``mode="grid"``.  All three
        dimensions span from the scene bounding-box minimum to maximum.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Candidate positions in world-space metres.

    Raises
    ------
    ValueError
        If ``mode="user"`` but *user_positions* is ``None`` or has the
        wrong shape, or if *mode* is not recognised.

    Examples
    --------
    User-supplied positions::

        candidates = get_candidate_locations(
            scene,
            mode="user",
            user_positions=my_xyz_array,
        )

    Uniform 3-D grid (20 m x 20 m x 20 m spacing)::

        candidates = get_candidate_locations(
            scene,
            mode="grid",
            dx=20, dy=20, dz=20,
        )
    """
    if mode == "user":
        return _candidates_user(user_positions)
    elif mode == "grid":
        bbox = sionna_scene._scene.bbox()
        return _candidates_grid(
            np.array(bbox.min), np.array(bbox.max),
            dx, dy, dz,
        )
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Choose 'user' or 'grid'."
        )


def _candidates_user(user_positions: Optional[np.ndarray]) -> np.ndarray:
    """Return user-supplied positions after basic shape validation."""
    if user_positions is None:
        raise ValueError(
            "mode='user' requires user_positions to be provided."
        )
    positions = np.asarray(user_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"user_positions must have shape (N, 3), got {positions.shape}."
        )
    print(f"User candidates: {len(positions)} positions supplied.")
    return positions


def _candidates_grid(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Generate a uniform 3-D grid of candidates over the scene bounding box."""
    xs = np.arange(bbox_min[0], bbox_max[0] + dx * 0.5, dx)
    ys = np.arange(bbox_min[1], bbox_max[1] + dy * 0.5, dy)
    zs = np.arange(bbox_min[2], bbox_max[2] + dz * 0.5, dz)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    candidates = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    print(
        f"Grid candidates: {len(candidates)} nodes "
        f"({len(xs)} x {len(ys)} x {len(zs)}) "
        f"with spacing dx={dx} m, dy={dy} m, dz={dz} m."
    )
    return candidates


# ---------------------------------------------------------------------------
# Terrain height utilities (for reference tower post-processing)
# ---------------------------------------------------------------------------

def build_terrain_tree(
    sionna_scene,
    terrain_object_name: str,
) -> Tuple[cKDTree, np.ndarray]:
    """Build a KD-tree over terrain vertex XY positions for height look-ups.

    This is used to snap real-world tower coordinates (which may only have
    XY information) to the correct Z height in the Sionna scene.  It is
    **not** required by the candidate selection or optimisation loop.

    Parameters
    ----------
    sionna_scene :
        Loaded Sionna scene (``merge_shapes=False``).
    terrain_object_name : str
        Key of the terrain/ground object in ``sionna_scene.objects``.
        For the built-in scenes this is ``"Terrain"`` (San Francisco) or
        ``"ground"`` (Florence).  Inspect ``sionna_scene.objects.keys()``
        to find the right name for a custom scene.

    Returns
    -------
    (cKDTree, np.ndarray)
        KD-tree for 2-D XY nearest-neighbour queries, and the full vertex
        array (shape M x 3).

    Raises
    ------
    KeyError
        If *terrain_object_name* is not present in the scene.
    """
    if terrain_object_name not in sionna_scene.objects:
        available = list(sionna_scene.objects.keys())
        raise KeyError(
            f"Terrain object '{terrain_object_name}' not found in scene. "
            f"Available objects: {available}"
        )
    mi = sionna_scene.objects[terrain_object_name].mi_mesh
    verts = np.array(
        [mi.vertex_position(i) for i in range(mi.vertex_count())]
    )[:, :, 0]
    return cKDTree(verts[:, :2]), verts


def apply_tower_height(
    towers: np.ndarray,
    terrain_tree: cKDTree,
    terrain_verts: np.ndarray,
    tower_height: float = 20.0,
) -> np.ndarray:
    """Snap tower XY positions to terrain height and add a mast offset.

    Parameters
    ----------
    towers : np.ndarray, shape (N, 2) or (N, 3)
        Tower XY (or XYZ) locations.  The Z column is ignored and replaced.
    terrain_tree : scipy.spatial.cKDTree
        KD-tree from :func:`build_terrain_tree`.
    terrain_verts : np.ndarray, shape (M, 3)
        Full terrain vertex array (x, y, z).
    tower_height : float
        Mast height above local terrain in metres.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Towers with Z = z_terrain + tower_height.
    """
    _, idx = terrain_tree.query(towers[:, :2])
    z_base = terrain_verts[idx, 2]
    return np.column_stack([towers[:, :2], z_base + tower_height])


# ---------------------------------------------------------------------------
# Basis function (radio map) computation
# ---------------------------------------------------------------------------

def compute_basis_functions(
    sionna_scene,
    candidates: np.ndarray,
    output_folder: str | Path,
    power_dbm: float = 40.0,
    bandwidth_hz: float = 10e6,
    snr_gap_gamma: float = 2.0,
    max_depth: int = 5,
    samples_per_tx: int = int(1e6),
    frequency_hz: float = 1.8e9,
    start_idx: int = 0,
) -> None:
    """Compute and save one rate-map ``.npy`` per candidate transmitter.

    For each candidate position, a single isotropic transmitter is placed,
    Sionna's ``RadioMapSolver`` is run, and the Shannon achievable rate
    R(y) = B * log2(1 + SINR(y) / Gamma) is saved as a 2-D array.

    Results are written incrementally after each candidate so the run can be
    safely interrupted and resumed with :func:`compute_basis_functions_resume`.

    Works with any Sionna scene.

    Parameters
    ----------
    sionna_scene :
        Loaded scene.  Should be loaded with ``merge_shapes=True`` for
        ray-tracing.
    candidates : np.ndarray, shape (N, 3)
        Candidate transmitter positions in world-space metres.
    output_folder : str or Path
        Root output folder.  A ``Sionna/`` sub-directory is created
        containing:

        * ``0_Coordinates.txt`` -- one line per candidate: ``i, x, y, z``.
        * ``{i}.npy``           -- 2-D rate map for candidate *i* (bps).

    power_dbm : float
        Transmit power in dBm.
    bandwidth_hz : float
        System bandwidth in Hz.
    snr_gap_gamma : float
        SNR gap Gamma for the Shannon capacity formula.
    max_depth : int
        Ray-tracing recursion depth.
    samples_per_tx : int
        Monte Carlo ray samples per transmitter per call.
    frequency_hz : float
        Carrier frequency in Hz.
    start_idx : int
        Index of the first candidate to process.  Set automatically by
        :func:`compute_basis_functions_resume`; leave at 0 for a fresh run.
    """
    output_folder = Path(output_folder)
    sionna_dir = output_folder / "Sionna"
    sionna_dir.mkdir(parents=True, exist_ok=True)

    # Write the coordinate manifest on a fresh run
    if start_idx == 0:
        with open(sionna_dir / "0_Coordinates.txt", "w") as fh:
            for i, (x, y, z) in enumerate(candidates):
                fh.write(f"{i}, {x}, {y}, {z}\n")

    sionna_scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.0, horizontal_spacing=0.0,
        pattern="iso", polarization="V",
    )
    sionna_scene.frequency = frequency_hz

    solver = RadioMapSolver()
    tx = Transmitter(name="tx", position=[0.0, 0.0, 0.0], power_dbm=power_dbm)
    sionna_scene.add(tx)

    noise_watts = 10 ** ((-174 + 10 * np.log10(bandwidth_hz)) / 10)
    p_tx_watts = 10 ** ((power_dbm - 30) / 10)

    for i in tqdm(range(start_idx, len(candidates)), desc="Computing basis functions"):
        tx.position = candidates[i].tolist()

        rm = solver(
            scene=sionna_scene,
            cell_size=[1.0, 1.0],
            max_depth=max_depth,
            samples_per_tx=samples_per_tx,
        )

        pg = np.maximum(rm.path_gain.numpy(), 1e-30)
        p_rx = p_tx_watts * pg
        sinr = p_rx[0] / noise_watts
        rate = bandwidth_hz * np.log2(1.0 + sinr / snr_gap_gamma)
        np.save(sionna_dir / f"{i}.npy", rate)


def compute_basis_functions_resume(
    output_folder: str | Path,
    sionna_scene,
    **kwargs,
) -> None:
    """Resume an interrupted :func:`compute_basis_functions` run.

    Reads the existing ``0_Coordinates.txt`` to recover candidate positions,
    finds the highest-numbered completed ``.npy`` file, and restarts from
    the next index.

    Parameters
    ----------
    output_folder : str or Path
        Same folder passed to the original :func:`compute_basis_functions`.
    sionna_scene :
        Freshly loaded scene.  GPU state is not preserved across restarts.
    **kwargs
        Forwarded verbatim to :func:`compute_basis_functions`.
    """
    output_folder = Path(output_folder)
    sionna_dir = output_folder / "Sionna"

    with open(sionna_dir / "0_Coordinates.txt") as fh:
        candidates = np.array(
            [[float(v) for v in line.strip().split(", ")[1:]] for line in fh]
        )

    existing = [int(f.stem) for f in sionna_dir.glob("*.npy") if f.stem.isdigit()]
    start_idx = max(existing) + 1 if existing else 0
    print(f"Resuming from candidate index {start_idx} / {len(candidates)}.")

    compute_basis_functions(
        sionna_scene, candidates, output_folder, start_idx=start_idx, **kwargs
    )


# ---------------------------------------------------------------------------
# Diagnostic visualisation
# ---------------------------------------------------------------------------

def candidate_locations_plot(
    candidates: np.ndarray,
    save_path: str | Path,
) -> None:
    """Save an interactive 3-D scatter of candidate locations.

    Saves as an HTML file if *save_path* ends in ``.html``, otherwise as a
    static image (requires the ``kaleido`` package).

    Parameters
    ----------
    candidates : np.ndarray, shape (N, 3)
        Candidate positions to visualise.
    save_path : str or Path
        Output file path.
    """
    fig = go.Figure(
        go.Scatter3d(
            x=candidates[:, 0],
            y=candidates[:, 1],
            z=candidates[:, 2],
            mode="markers",
            marker=dict(
                size=2, opacity=0.7,
                color=candidates[:, 2], colorscale="Viridis",
                colorbar=dict(title="Z (m)"),
            ),
        )
    )
    fig.update_layout(
        title=f"Candidate Transmitter Locations ({len(candidates)} sites)",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix == ".html":
        fig.write_html(str(save_path))
    else:
        fig.write_image(str(save_path))
    print(f"Candidate plot -> {save_path}")


def basis_function_gif(
    sionna_dir: str | Path,
    output_path: str = "basis_functions.gif",
) -> None:
    """Render all saved rate maps as an animated GIF for quick inspection.

    Parameters
    ----------
    sionna_dir : str or Path
        The ``Sionna/`` sub-directory produced by :func:`compute_basis_functions`.
    output_path : str
        Output GIF file path.
    """
    sionna_dir = Path(sionna_dir)
    indices = sorted(int(f.stem) for f in sionna_dir.glob("*.npy") if f.stem.isdigit())
    tmp = Path("_tmp_gif_frames")
    tmp.mkdir(exist_ok=True)
    try:
        for i in tqdm(indices, desc="Rendering frames"):
            A = np.load(sionna_dir / f"{i}.npy")
            plt.clf()
            plt.imshow(A, cmap="viridis")
            plt.title(f"Candidate {i}")
            plt.colorbar(label="Rate (bps)")
            plt.savefig(tmp / f"{i:06d}.png", dpi=80)
        frames = [imageio.imread(tmp / f"{i:06d}.png") for i in indices]
        imageio.mimsave(output_path, frames, duration=0.1, loop=0)
        print(f"GIF saved -> {output_path}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
