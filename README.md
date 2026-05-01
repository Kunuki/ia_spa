# IA-SPA: Interference-Aware Submodular Placement Algorithm

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Sionna](https://img.shields.io/badge/Sionna-RT-green)](https://nvlabs.github.io/sionna/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of the paper:

> **Optimal Transmitter Placement in Realistic Urban Environments**  
> Lukas Taus, Richard Tsai, Jeffrey G. Andrews  
> Oden Institute for Computational Engineering and Sciences &  
> 6G@UT Wireless Networking and Communications Group,  
> The University of Texas at Austin  
> *Submitted to IEEE Transactions on Wireless Communications*  
> arXiv:2604.28153 [cs.IT] — https://arxiv.org/abs/2604.28153

---

## Overview

IA-SPA is a mathematically rigorous framework for placing wireless transmitters
(e.g. cellular base stations) in site-specific 3-D urban environments.  It
combines:

- **Sionna RT ray tracing** for high-fidelity, physically accurate path-gain fields
- **Submodular greedy optimisation** with formal approximation guarantees (≥ 63% of
  the global optimum for equal budget, Theorem III.2)
- **An interference-aware quality functional S(T)** that balances coverage,
  capacity, and interference

Applied to San Francisco and Florence, IA-SPA achieves:

| Scenario | Mean rate gain | Edge rate (5th pct) gain |
|---|---|---|
| AT&T (SF, P_MAX) | **+71%** | **+135%** |
| T-Mobile (SF, P_MAX) | **+215%** | **+758%** |
| Iliad (Florence, exclusion zone) | +30% | **+56%** |

---

## Repository Structure

```
ia_spa/                 # Core Python package
├── __init__.py
├── optimizer.py        # TowerOptimizer class and run_greedy()
├── basis_functions.py  # SDF, candidate extraction, Sionna ray-trace loop
└── metrics.py          # SINR / rate / interference evaluation

scripts/                # Command-line entry points
├── compute_basis.py    # Pre-compute ray-traced basis functions (any scene)
├── run_optimizer.py    # Run the greedy IA-SPA loop
└── evaluate.py         # Evaluate results: load positions, compute and save metrics

notebooks/
├── 01_algorithm_overview.ipynb   # Self-contained intro (no GPU needed)
├── 02_san_francisco.ipynb        # Full SF pipeline
└── 03_florence_advanced.ipynb    # Exclusionary zones + incremental deployment

config/
└── config.yaml         # All simulation parameters

data/                   # (not tracked by git — see Data section below)
├── TowerData/          # Reference carrier tower positions (.npy)
├── MapData/            # SDF grids
├── BasisFunctions/     # Pre-computed ray-traced rate maps
└── Results/            # Greedy optimiser outputs
```

---

## Installation

### Requirements

- Python 3.10+
- NVIDIA GPU with ≥ 16 GB VRAM (A100 recommended)
- CUDA 12.x, cuDNN 8.x

### Steps

```bash
git clone https://github.com/<your-handle>/ia-spa.git
cd ia-spa

# Install Sionna (follow official instructions for your CUDA version)
pip install sionna

# Install remaining dependencies
pip install -r requirements.txt

# Install the ia_spa package in editable mode
pip install -e .
```

---

## Quick Start

### 1 — Configure

The repository ships with `config/config.yaml` as a ready-to-use template.
Edit it directly, or copy it elsewhere and pass the path via `--config`:

```bash
cp config/config.yaml my_config.yaml   # optional — scripts default to config/config.yaml
```

Key parameters in `config/config.yaml`:

| Key | Default | Description |
|---|---|---|
| `scene.frequency_hz` | 1.8e9 | Carrier frequency (Hz) |
| `transmitter.power_dbm` | 40.0 | TX power (dBm) |
| `link.bandwidth_hz` | 10e6 | System bandwidth (Hz) |
| `optimizer.aggregation` | `max` | `max` (P_MAX) or `sum` (P_SUM) |
| `optimizer.n_iter` | 20 | Number of transmitters to place |

### 2 — Pre-compute Basis Functions (one-time, GPU required)

```bash
# San Francisco
python -m scripts.compute_basis \
    --scene san_francisco \
    --output data/BasisFunctions/SF

# Florence
python -m scripts.compute_basis \
    --scene florence \
    --output data/BasisFunctions/FL

# Resume an interrupted run
python -m scripts.compute_basis \
    --scene san_francisco \
    --output data/BasisFunctions/SF \
    --resume
```

Grid spacing and all physical parameters are read from `config/config.yaml`
and can be overridden per-run with `--dx`, `--dy`, `--dz`, or `--config`.

### 3 — Run the Optimiser

```bash
python -m scripts.run_optimizer \
    --basis  data/BasisFunctions/SF \
    --results data/Results/SF_max \
    --n-iter 20 \
    --aggregation max
```

**Warm-start from pre-placed towers (incremental deployment):**

```bash
python -m scripts.run_optimizer \
    --basis  data/BasisFunctions/FL \
    --results data/Results/FL_incremental \
    --n-iter 20 \
    --preplaced-towers data/TowerData/FL_Iliad.npy
```

### 4 — Evaluate Results

```bash
python -m scripts.evaluate --results data/Results/SF_max
```

The scene is read automatically from `data/Results/SF_max/scene.txt` (written
by `run_optimizer.py`).  Pass `--scene` to override:

```bash
python -m scripts.evaluate \
    --results data/Results/MyScene_max \
    --scene path/to/my_scene.xml
```

SINR, rate, and interference maps (`.npy`) plus a printed summary (mean rate,
5th-percentile rate, mean SINR) are saved to `data/Results/SF_max_Processed/`.

**Optional flags:**

| Flag | Description |
|---|---|
| `--plot` | Save a two-panel PNG (`*_map.png`) showing the rate map and SINR map with tower locations overlaid |
| `--n-towers N` | Evaluate only the first N towers from `Locations.txt` |

```bash
python -m scripts.evaluate --results data/Results/SF_max --plot
```

---

## Notebooks

| Notebook | Description | GPU needed? |
|---|---|---|
| `01_algorithm_overview.ipynb` | Math and synthetic demo | ❌ No |
| `02_san_francisco.ipynb` | Full SF workflow | ✅ Yes |
| `03_florence_advanced.ipynb` | Exclusionary zones, incremental | ✅ Yes |

---

## Data

Reference tower locations are sourced from:

- **San Francisco** — [City and County of San Francisco Open Data](https://data.sfgov.org/Geographic-Locations-and-Boundaries/Existing-Commercial-Wireless-Telecommunication-Ser/aa26-h926)
- **Florence (Iliad, TIM, Vodafone, WindTre)** — [OpenCellID](https://opencellid.org/)

Place the processed `.npy` files in `data/TowerData/`:

```
data/TowerData/
├── SF_ATT.npy
├── SF_TMobile.npy
├── SF_ATT_TMobile.npy
├── FL_Iliad.npy
├── FL_TIM.npy
├── FL_Vodafone.npy
└── FL_WindTre.npy
```

Each file contains an (N, 3) array of (x, y, z) coordinates in the Sionna
scene's local coordinate system (metres).

---

## Algorithm Summary

```
T = []  (or T = T_fixed for incremental deployment)
while |T| < k:
    for each candidate x ∈ X:
        compute G(x|T) = S(T ∪ {x}) − S(T)
    select x* = argmax G(x|T)
    T ← T ∪ {x*}
```

where

$$S(T) = \mathbb{E}_{y \sim f}\left[\bar{W}\\left(\max_{t \in T} P(y,t)\right)\right], \quad \bar{W}(x) = \frac{x}{x+c}$$

**Guarantee (Theorem III.2):**  With $n = k$ iterations and $\epsilon = 0$,

$$S(T_n) \geq (1 - e^{-1}) S(T^*_k) \approx 0.632 \cdot S(T^*_k)$$

---

## Citation

If you use this code, please cite:

```bibtex
@article{taus2026iaspa,
  title   = {Optimal Transmitter Placement in Realistic Urban Environments},
  author  = {Taus, Lukas and Tsai, Richard and Andrews, Jeffrey G.},
  journal = {arXiv preprint arXiv:2604.28153},
  year    = {2026},
  note    = {Submitted to IEEE Transactions on Wireless Communications},
  url     = {https://arxiv.org/abs/2604.28153},
}
```

---

## License

This project is released under the [MIT License](LICENSE).
