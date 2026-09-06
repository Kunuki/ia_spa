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

The repository has two entry points:

1. **[Radio maps → tower locations](#quick-start-radio-maps--tower-locations)** —
   the placement algorithm on its own.  Bring your own radio maps from any
   source; no Sionna, no GPU, no 3-D scene required.
2. **[The full pipeline](#full-pipeline-ray-tracing-from-a-3-d-scene)** — ray-trace
   the radio maps for a scene with Sionna RT, place the transmitters, and
   evaluate the resulting SINR / rate / interference maps.

---

## Installation

```bash
git clone https://github.com/Kunuki/ia_spa.git
cd ia_spa
pip install -e .
```

That is everything the placement algorithm needs (NumPy, SciPy, tqdm, PyYAML)
and it installs the `ia-spa` command.

For ray tracing, evaluation and the notebooks, add the optional extras and
Sionna:

```bash
pip install -e ".[viz]"
pip install sionna          # follow the official instructions for your CUDA version
```

Ray tracing needs an NVIDIA GPU with ≥ 16 GB VRAM (A100 recommended), CUDA 12.x
and cuDNN 8.x.  The placement algorithm itself runs on any CPU.

---

## Quick Start: radio maps → tower locations

This is the shortest path from data to an answer.  You supply

- **a file of radio maps** — one map per candidate transmitter site, giving the
  achievable rate (or received power) at every point of the service area when
  *only that one transmitter* is switched on, and
- **a file of the candidate locations** — the (x, y, z) coordinate of each of
  those sites, in the same order as the maps;

and you get back **the locations of the towers to build**.

```bash
ia-spa --radio-maps radio_maps.npy --locations sites.csv --n-towers 15
```

```
40 candidate sites, radio maps of shape (60, 60) (3,600 cells), in memory; median non-zero value 3.5e+06.
IA-SPA placement: 15 transmitter(s) chosen from 40 candidate site(s)
  aggregation = max   utility_c = 1e+08
  S(T) = 0.000000 -> 0.497069 (+0.497069)
  #   candidate            x           y           z       gain
  1   17              26.00       30.00       15.00   0.123705
  2   26              42.00       16.00       15.00   0.053943
  3   28              42.00       44.00       15.00   0.050599
  ...
Tower locations written to 'towers.csv'.
```

`towers.csv` holds the chosen sites in selection order, with the candidate
index they came from and the marginal gain each one contributed:

```csv
candidate_index,x,y,z,marginal_gain
17,26.000000,30.000000,15.000000,0.12370528
26,42.000000,16.000000,15.000000,0.05394323
28,42.000000,44.000000,15.000000,0.050599181
```

The same thing from Python:

```python
from ia_spa import place_towers

result = place_towers("radio_maps.npy", "sites.csv", n_towers=15)

result.positions      # (15, 3) array of chosen tower coordinates
result.indices        # which candidate each one was
result.gains          # marginal gain G(x|T) at each step
result.quality        # S(T) after 0, 1, ... 15 placements
print(result.summary())
result.save("towers.csv")
```

NumPy arrays work just as well as file paths, so the algorithm drops into an
existing pipeline without touching the disk:

```python
result = place_towers(radio_maps, site_coordinates, n_towers=15)
```

### Input formats

**Radio maps** (`--radio-maps` / `-m`) — one non-negative map per candidate,
all on the same grid, in **linear** units (not dB):

| Format | Layout |
|---|---|
| `.npy` | `(N, H, W)` stack, or `(N, P)` if the maps are already flattened. Memory-mapped, so files larger than RAM are fine. |
| `.npz` | One stacked array (key `radio_maps`, `maps`, `basis` or `arr_0`), or one 2-D array per candidate under numeric keys `"0"`, `"1"`, … |
| `.csv` `.txt` `.tsv` `.dat` | N rows × P columns of text; each row is one flattened map. |
| directory | One `{i}.npy` per candidate (`0.npy`, `1.npy`, …). A `Sionna/` sub-directory and a `0_Coordinates.txt` manifest are picked up automatically, so a folder written by `scripts/compute_basis.py` can be passed directly and `--locations` becomes optional. |

**Locations** (`--locations` / `-l`) — an `(N, 3)` table of (x, y, z) in the same
order as the maps, as `.npy`, `.csv` or `.txt`.  A header line is skipped, `(N, 2)`
(x, y) input gets `z = 0`, and a leading integer index column — the
`i, x, y, z` format of `0_Coordinates.txt` — is recognised and dropped.

**Output** (`--output` / `-o`, default `towers.csv`) — `.csv` for the table
above, `.txt` for bare `x, y, z` lines, `.npy` for an `(N, 3)` array, or a
directory to receive both `towers.csv` and a `summary.json` of the whole run.

### Options

| Flag | Default | Description |
|---|---|---|
| `-n`, `--n-towers` | 10 | How many transmitters to place. |
| `-a`, `--aggregation` | `max` | `max` (P_MAX — each user associates with the strongest base station; realistic) or `sum` (P_SUM — the idealised upper bound of the paper). |
| `-c`, `--utility-c` | `1e8` | Saturation constant *c* of W̄(x) = x/(x+c), **in the units of your radio maps**. The default assumes rate in bps, so 1e8 = 100 Mbps. |
| `--fixed-towers` | — | Coordinates of transmitters that already exist. They are snapped to their nearest candidate sites and held fixed, so the run returns the best *n* **additions** to an existing deployment (incremental deployment, Section II-C). |
| `--density` | uniform | Per-cell spatial priority weights — e.g. a population raster — with the same shape as one radio map. Normalised internally. |
| `--allow-repeats` / `--no-repeats` | per aggregation | Whether one site may be chosen twice. Off under `max`, where a repeat provably adds nothing; on under `sum`, where co-located power does add up. |
| `--check-only` | — | Load and validate the inputs, print a summary, and exit without computing. |
| `-q`, `--quiet` | — | Suppress the progress bar and per-iteration output. |

`python -m ia_spa …` and `python -m scripts.place_towers …` run the same
command without the installed console script.

### Getting the units right

The utility W̄(x) = x/(x+c) is what makes the placement interference-aware: it
saturates, so a second transmitter covering an already-served area is worth less
than one covering a dead zone.  That only works when the typical served value of
your maps is within a few orders of magnitude of `c`.  Two rules follow:

- Radio maps must be **linear and non-negative** — path gain or rate, never dB.
  Negative values are rejected with an explanatory error.
- `--utility-c` must be in the **same units as the maps**.  If your maps are
  received power in watts rather than rate in bps, set `-c` accordingly.

A run whose scales are badly mismatched prints a warning saying which way it is
off; `--check-only` reports the same without computing anything.

### What the algorithm does

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

```math
S(T_n) \ge (1 - e^{-1})\, S(T^{\star}_k) \approx 0.632\, S(T^{\star}_k)
```

Each iteration makes a single pass over the radio maps, in bounded-memory
chunks, so the cost is O(k · N · cells) in time and independent of k in memory.

---

## Full pipeline: ray tracing from a 3-D scene

Use this when you want IA-SPA to generate the radio maps itself, from a Sionna
scene.  Requires Sionna and a GPU.

### 1 — Configure

`config/config.yaml` ships as a ready-to-use template.  Edit it in place, or
copy it and pass `--config`.

| Key | Default | Description |
|---|---|---|
| `scene.frequency_hz` | 1.8e9 | Carrier frequency (Hz) |
| `transmitter.power_dbm` | 40.0 | TX power (dBm) |
| `transmitter.tower_height_m` | 20.0 | Mast height above local terrain (m) |
| `link.bandwidth_hz` | 10e6 | System bandwidth (Hz) |
| `radio_map.cell_size_m` | 1.0 | Radio-map pixel resolution (m) |
| `optimizer.aggregation` | `max` | `max` (P_MAX) or `sum` (P_SUM) |
| `optimizer.n_iter` | 20 | Number of transmitters to place |

### 2 — Pre-compute the radio maps (one-time, GPU required)

```bash
# San Francisco
python -m scripts.compute_basis --scene san_francisco --output data/BasisFunctions/SF

# Florence
python -m scripts.compute_basis --scene florence --output data/BasisFunctions/FL

# Resume an interrupted run
python -m scripts.compute_basis --scene san_francisco --output data/BasisFunctions/SF --resume
```

Grid spacing and all physical parameters come from `config/config.yaml` and can
be overridden per run with `--dx`, `--dy`, `--dz`, `--map-height` or `--config`.
Add `-y` to overwrite an existing output folder without the confirmation prompt.

Every candidate must land on the same radio-map grid; the script fails fast with
a clear error if one does not.

### 3 — Run the optimiser

```bash
python -m scripts.run_optimizer \
    --basis   data/BasisFunctions/SF \
    --results data/Results/SF_max \
    --n-iter 20 \
    --aggregation max
```

**Warm-start from pre-placed towers (incremental deployment):**

```bash
python -m scripts.run_optimizer \
    --basis   data/BasisFunctions/FL \
    --results data/Results/FL_incremental \
    --n-iter 20 \
    --preplaced-towers data/TowerData/FL_Iliad.npy
```

`run_optimizer` also accepts `--radio-maps` / `--locations` instead of `--basis`,
for radio maps that did not come from `compute_basis.py`.  It writes
`Locations.txt`, `towers.csv`, `u.npy`, one `Gain_Function_k.npy` per iteration,
and a `summary.json` of the run.

### 4 — Evaluate the result

```bash
python -m scripts.evaluate --results data/Results/SF_max
```

The scene is read automatically from `data/Results/SF_max/scene.txt` (written by
`run_optimizer.py`).  Pass `--scene` to override:

```bash
python -m scripts.evaluate --results data/Results/MyScene_max --scene path/to/my_scene.xml
```

SINR, rate and interference maps (`.npy`) plus a printed summary (mean rate,
5th-percentile rate, mean SINR) are saved to `data/Results/SF_max_Processed/`.

| Flag | Description |
|---|---|
| `--plot` | Save a two-panel PNG (`*_map.png`) of the rate and SINR maps with tower locations overlaid |
| `--n-towers N` | Evaluate only the first N towers from `Locations.txt` |

---

## Repository Structure

```
ia_spa/                 # Core Python package
├── placement.py        # place_towers() -- radio maps in, tower locations out
├── optimizer.py        # TowerOptimizer, greedy_select(), run_greedy(), S(T) and G(x|T)
├── data.py             # Radio-map / location loading, validation and saving
├── cli.py              # The `ia-spa` command
├── basis_functions.py  # Candidate extraction and the Sionna ray-trace loop   [Sionna]
├── metrics.py          # SINR / rate / interference evaluation                [Sionna]
└── scenes.py           # Built-in and custom Sionna scene loading             [Sionna]

scripts/                # Command-line entry points
├── place_towers.py     # Same as the `ia-spa` command
├── compute_basis.py    # Pre-compute ray-traced radio maps (any scene)
├── run_optimizer.py    # Run the greedy IA-SPA loop, with the on-disk run layout
└── evaluate.py         # Evaluate results: load positions, compute and save metrics

tests/
├── test_placement.py   # Radio maps -> towers: I/O, validation, accuracy, CLI
└── test_optimizer.py   # Utility function, candidate selection, greedy loop

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

## Notebooks

| Notebook | Description | GPU needed? |
|---|---|---|
| `01_algorithm_overview.ipynb` | Math and synthetic demo | ❌ No |
| `02_san_francisco.ipynb` | Full SF workflow | ✅ Yes |
| `03_florence_advanced.ipynb` | Exclusionary zones, incremental | ✅ Yes |

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/                      # unit tests, no GPU or Sionna needed
SIONNA_INTEGRATION=1 pytest tests/ # adds the Sionna scene integration tests
```

The suite checks the greedy loop three independent ways: against a naive
reference implementation of S(T) and G(x|T), against the exhaustive optimum on
small instances (including the (1 − 1/e) bound of Theorem III.2), and against
hand-built instances whose answer is known by construction.  It also covers
every input format, the validation errors, and the command-line interface.

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
