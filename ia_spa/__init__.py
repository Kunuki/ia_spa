"""
IA-SPA: Interference-Aware Submodular Placement Algorithm
=========================================================

A framework for optimal wireless transmitter placement in realistic 3-D urban
environments, using Sionna RT ray tracing and submodular greedy optimisation.

Quick start
-----------
Radio maps in, tower locations out -- no Sionna, no GPU, no scene needed::

    from ia_spa import place_towers

    result = place_towers("radio_maps.npy", "locations.csv", n_towers=15)
    print(result.positions)          # (15, 3) chosen tower coordinates
    result.save("towers.csv")

The same from a terminal::

    ia-spa --radio-maps radio_maps.npy --locations locations.csv -n 15

Full pipeline
-------------
``ia_spa.basis_functions`` ray-traces the per-candidate radio maps with
Sionna RT, and ``ia_spa.metrics`` evaluates a finished placement (SINR, rate
and interference maps).  Both require Sionna and a GPU; the placement
algorithm itself does not.

Reference
---------
L. Taus, R. Tsai, and J. G. Andrews, "Optimal Transmitter Placement in
Realistic Urban Environments," submitted to IEEE Transactions on Wireless
Communications, arXiv:2604.28153 [cs.IT], Apr. 2026.
https://arxiv.org/abs/2604.28153
"""

__version__ = "0.2.0"

from ia_spa.data import (
    RadioMapSet,
    load_locations,
    load_radio_maps,
    save_positions,
)
from ia_spa.optimizer import (
    PlacementResult,
    TowerOptimizer,
    greedy_select,
    run_greedy,
    w_bar,
)
from ia_spa.placement import check_utility_scale, place_towers

__all__ = [
    # High-level entry point
    "place_towers",
    "PlacementResult",
    # Optimiser
    "TowerOptimizer",
    "greedy_select",
    "run_greedy",
    "w_bar",
    "check_utility_scale",
    # Data handling
    "RadioMapSet",
    "load_radio_maps",
    "load_locations",
    "save_positions",
    # Sionna-backed helpers (imported lazily; see __getattr__ below)
    "apply_tower_height",
    "basis_function_gif",
    "build_terrain_tree",
    "candidate_locations_plot",
    "compute_basis_functions",
    "compute_basis_functions_resume",
    "get_candidate_locations",
    "compute_metrics",
    "evaluate_and_save",
    "load_greedy_positions",
    "plot_evaluation",
]

# The ray-tracing and evaluation helpers pull in matplotlib, plotly and
# scipy, and are only needed for the Sionna pipeline.  Import them on first
# use so that `from ia_spa import place_towers` stays fast and dependency-light.
_LAZY_MODULES = {
    "apply_tower_height": "basis_functions",
    "basis_function_gif": "basis_functions",
    "build_terrain_tree": "basis_functions",
    "candidate_locations_plot": "basis_functions",
    "compute_basis_functions": "basis_functions",
    "compute_basis_functions_resume": "basis_functions",
    "get_candidate_locations": "basis_functions",
    "compute_metrics": "metrics",
    "evaluate_and_save": "metrics",
    "load_greedy_positions": "metrics",
    "plot_evaluation": "metrics",
}


def __getattr__(name):
    """Lazily resolve the Sionna-pipeline helpers (PEP 562)."""
    module_name = _LAZY_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'ia_spa' has no attribute '{name}'")
    import importlib

    module = importlib.import_module(f"ia_spa.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
