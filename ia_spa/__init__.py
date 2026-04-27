"""
IA-SPA: Interference-Aware Submodular Placement Algorithm
=========================================================

A framework for optimal wireless transmitter placement in realistic 3-D urban
environments, using Sionna RT ray tracing and submodular greedy optimisation.

Works with any scene loadable via ``sionna.rt.load_scene``, or with any
Sionna scene object already in memory.

Reference
---------
Taus, Tsai, Andrews. "Optimal Transmitter Placement in Realistic Urban
Environments." (2026).
"""

from ia_spa.optimizer import TowerOptimizer, run_greedy, w_bar
from ia_spa.basis_functions import (
    apply_tower_height,
    basis_function_gif,
    build_terrain_tree,
    candidate_locations_plot,
    compute_basis_functions,
    compute_basis_functions_resume,
    get_candidate_locations,
)
from ia_spa.metrics import (
    compute_metrics,
    evaluate_and_save,
    load_greedy_positions,
)

__all__ = [
    # Optimizer
    "TowerOptimizer",
    "run_greedy",
    "w_bar",
    # Candidate selection and basis functions
    "apply_tower_height",
    "basis_function_gif",
    "build_terrain_tree",
    "candidate_locations_plot",
    "compute_basis_functions",
    "compute_basis_functions_resume",
    "get_candidate_locations",
    # Metrics / evaluation
    "compute_metrics",
    "evaluate_and_save",
    "load_greedy_positions",
]
