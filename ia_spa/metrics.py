"""
ia_spa/metrics.py
-----------------
Post-processing utilities: load optimised or reference tower positions, run a
multi-height Sionna evaluation, and save SINR / rate / interference maps.

This corresponds to the ``data_formatter_*.py`` scripts, refactored into a
single scene-agnostic function with explicit parameters rather than relying on
a fixed ``config.yaml`` path.  The ``main`` entry point in
``scripts/evaluate.py`` calls these helpers after reading the config.

Theory
------
For a given transmitter set T the SINR at receiver y is

    SINR(y, T) = max_{t∈T} P(y,t) / (Σ_{t∈T} P(y,t) − max_{t∈T} P(y,t) + σ²)

The Shannon achievable rate is then R(y) = B·log₂(1 + SINR(y,T)/Γ).

All metrics are averaged linearly over the receiver heights specified in the
config before being saved to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ia_spa.data import load_locations


def compute_metrics(
    sionna_scene,
    positions: np.ndarray,
    receiver_heights: Sequence[float] = (1.5, 5.0, 10.0),
    power_dbm: float = 40.0,
    bandwidth_hz: float = 10e6,
    snr_gap_gamma: float = 2.0,
    max_depth: int = 10,
    samples_per_tx: int = int(1e6),
    cell_size_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate SINR, rate, and interference for a transmitter configuration.

    All transmitters in *sionna_scene* are replaced by those in *positions*
    before solving.  Metrics are computed at each height in
    *receiver_heights* and averaged in linear scale.

    Parameters
    ----------
    sionna_scene :
        Loaded and configured Sionna scene (frequency and tx_array must be
        set before calling this function).
    positions : np.ndarray, shape (N, 3)
        World-space (x, y, z) of the N transmitters to evaluate.
    receiver_heights : sequence of float
        Receiver heights (m AGL) at which radio maps are computed and
        subsequently averaged.
    power_dbm : float
        Transmit power per BS in dBm.
    bandwidth_hz : float
        System bandwidth in Hz.
    snr_gap_gamma : float
        Shannon SNR gap Γ.
    max_depth : int
        Ray-tracing recursion depth.
    samples_per_tx : int
        Monte Carlo ray samples per transmitter.
    cell_size_m : float
        Radio-map pixel resolution in metres.

    Returns
    -------
    avg_sinr : np.ndarray
        Height-averaged SINR map (linear scale).
    avg_rate : np.ndarray
        Height-averaged achievable rate map (bps).
    interf_dbm : np.ndarray
        Height-averaged interference power map (dBm).
    """
    from sionna.rt import RadioMapSolver, Transmitter  # GPU required

    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    if len(positions) == 0:
        raise ValueError("compute_metrics needs at least one transmitter position.")

    # Remove any existing transmitters and add the new set
    for tx_name in list(sionna_scene.transmitters.keys()):
        sionna_scene.remove(tx_name)

    for i, pos in enumerate(positions):
        sionna_scene.add(
            Transmitter(name=f"tx_{i}", position=pos.tolist(), power_dbm=power_dbm)
        )

    # Determine map geometry from scene bounding box
    bbox = sionna_scene._scene.bbox()
    center_xy = [
        0.5 * (bbox.min[0] + bbox.max[0]),
        0.5 * (bbox.min[1] + bbox.max[1]),
    ]
    size_xy = [bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1]]

    solver = RadioMapSolver()
    noise_watts = 10 ** ((-174 + 10 * np.log10(bandwidth_hz)) / 10)
    p_tx_watts = 10 ** ((power_dbm - 30) / 10)

    all_sinr, all_rate, all_interf = [], [], []

    for height in receiver_heights:
        rm = solver(
            scene=sionna_scene,
            center=center_xy + [height],
            orientation=[0.0, 0.0, 0.0],
            size=size_xy,
            cell_size=[cell_size_m, cell_size_m],
            max_depth=max_depth,
            samples_per_tx=samples_per_tx,
        )

        pg = np.maximum(rm.path_gain.numpy(), 1e-30)
        p_rx = p_tx_watts * pg  # shape: (n_tx, H, W)

        serving = np.max(p_rx, axis=0)
        total = np.sum(p_rx, axis=0)
        interf_w = np.maximum(total - serving, 1e-20)

        sinr = serving / (interf_w + noise_watts)
        rate = bandwidth_hz * np.log2(1.0 + sinr / snr_gap_gamma)

        all_sinr.append(sinr)
        all_rate.append(rate)
        all_interf.append(interf_w)

    avg_sinr = np.mean(np.stack(all_sinr), axis=0)
    avg_rate = np.mean(np.stack(all_rate), axis=0)
    avg_interf_w = np.mean(np.stack(all_interf), axis=0)
    interf_dbm = 10 * np.log10(avg_interf_w) + 30

    return avg_sinr, avg_rate, interf_dbm


def load_greedy_positions(results_folder: str | Path) -> np.ndarray:
    """Load tower (x, y, z) positions written by ``run_greedy``.

    Parameters
    ----------
    results_folder : str or Path
        Folder containing ``Locations.txt``.

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    return load_locations(Path(results_folder) / "Locations.txt")


def evaluate_and_save(
    sionna_scene,
    name: str,
    positions: np.ndarray,
    save_dir: str | Path,
    receiver_heights: Sequence[float] = (1.5, 5.0, 10.0),
    power_dbm: float = 40.0,
    bandwidth_hz: float = 10e6,
    snr_gap_gamma: float = 2.0,
    cell_size_m: float = 1.0,
) -> None:
    """Compute metrics for *positions* and save ``{name}_{sinr,rate,interf,pos}.npy``.

    Parameters
    ----------
    sionna_scene :
        Configured Sionna scene.
    name : str
        Prefix for output filenames (e.g. ``"ATT_ref"`` or ``"TMobile_Greedy"``).
    positions : np.ndarray, shape (N, 3)
    save_dir : str or Path
        Output directory (created if absent).
    receiver_heights, power_dbm, bandwidth_hz, snr_gap_gamma, cell_size_m :
        Forwarded to :func:`compute_metrics`.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating '{name}' ({len(positions)} transmitters)…")
    sinr, rate, interf = compute_metrics(
        sionna_scene,
        positions,
        receiver_heights=receiver_heights,
        power_dbm=power_dbm,
        bandwidth_hz=bandwidth_hz,
        snr_gap_gamma=snr_gap_gamma,
        cell_size_m=cell_size_m,
    )

    np.save(save_dir / f"{name}_sinr.npy", sinr)
    np.save(save_dir / f"{name}_rate.npy", rate)
    np.save(save_dir / f"{name}_interf.npy", interf)
    np.save(save_dir / f"{name}_pos.npy", positions)
    print(
        f"  Mean rate: {rate.mean() / 1e6:.2f} Mbps  |  "
        f"5th pct: {np.percentile(rate, 5) / 1e6:.2f} Mbps  |  "
        f"Mean SINR: {10 * np.log10(sinr.mean()):.1f} dB"
    )


def plot_evaluation(
    rate: np.ndarray,
    sinr: np.ndarray,
    positions: np.ndarray,
    save_path: str | Path,
    scene_bbox: Optional[tuple] = None,
) -> None:
    """Save a two-panel figure: rate map and SINR map, both with tower scatter.

    Parameters
    ----------
    rate : np.ndarray, shape (H, W)
        Height-averaged achievable rate map in bps.
    sinr : np.ndarray, shape (H, W)
        Height-averaged SINR map (linear).
    positions : np.ndarray, shape (N, 3)
        World-space (x, y, z) of the placed transmitters.
    save_path : str or Path
        Output PNG file path.
    scene_bbox : ((x_min, y_min), (x_max, y_max)) or None
        Scene bounding box in metres.  Used to label axes.  When *None*,
        axes are in pixels.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if scene_bbox is not None:
        (x_min, y_min), (x_max, y_max) = scene_bbox
        extent = [x_min, x_max, y_min, y_max]
        xlabel, ylabel = "X (m)", "Y (m)"
        tx_x, tx_y = positions[:, 0], positions[:, 1]
    else:
        # Without a bounding box the axes are in pixels, and the world-space
        # tower coordinates cannot be placed on them, so they are omitted.
        extent = None
        xlabel, ylabel = "Pixel X", "Pixel Y"
        tx_x = tx_y = None

    # Rate map (Mbps)
    ax = axes[0]
    im0 = ax.imshow(
        rate.T / 1e6, origin="lower", aspect="equal",
        extent=extent, cmap="viridis",
    )
    if tx_x is not None:
        ax.scatter(tx_x, tx_y, c="red", s=40, marker="^", zorder=5, label="Towers")
        ax.legend(fontsize=8)
    ax.set_title("Achievable Rate (Mbps)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im0, ax=ax, label="Rate (Mbps)")

    # SINR map (dB)
    ax = axes[1]
    sinr_db = 10 * np.log10(np.maximum(sinr, 1e-10))
    im1 = ax.imshow(
        sinr_db.T, origin="lower", aspect="equal",
        extent=extent, cmap="plasma",
    )
    if tx_x is not None:
        ax.scatter(tx_x, tx_y, c="red", s=40, marker="^", zorder=5, label="Towers")
        ax.legend(fontsize=8)
    ax.set_title("SINR (dB)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im1, ax=ax, label="SINR (dB)")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {save_path}")
