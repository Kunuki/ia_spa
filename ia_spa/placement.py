"""
ia_spa/placement.py
-------------------
The one-call entry point to IA-SPA.

Given a set of radio maps -- one per candidate transmitter site -- and the
coordinates of those sites, :func:`place_towers` returns the subset of sites
that maximises the interference-aware network quality S(T), using the
submodular greedy algorithm of the paper.

    from ia_spa import place_towers

    result = place_towers("radio_maps.npy", "locations.csv", n_towers=15)
    print(result.positions)      # (15, 3) chosen tower coordinates
    result.save("towers.csv")

Everything is optional beyond those three arguments; see the docstring below
for warm starts, priority densities and the P_SUM aggregation variant.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ia_spa.data import (
    RadioMapSet,
    coerce_locations,
    load_locations,
    load_radio_maps,
)
from ia_spa.optimizer import PlacementResult, TowerOptimizer, greedy_select

__all__ = ["place_towers", "check_utility_scale"]

PathLike = Union[str, Path]

# W(x) = x/(x+c) is only informative when the radio-map values live within a
# few orders of magnitude of c.  Outside this band it degenerates to a
# constant (saturated) or to a linear function (no diminishing returns).
_SCALE_WARN_RATIO = 1e3


def check_utility_scale(
    maps: RadioMapSet, utility_c: float, n_sample: int = 8
) -> Optional[str]:
    """Warn when the saturation constant is mismatched to the radio-map units.

    The utility W(x) = x / (x + c) only rewards spreading coverage when the
    typical served value x is comparable to *c*.  If x >> c every served cell
    is already saturated; if x << c the utility is effectively linear and the
    greedy loop degenerates into maximising the mean field.  Both cases
    usually mean the maps and *c* are in different units.

    Returns
    -------
    str or None
        The warning message that was issued, or None if the scales are sane.
    """
    sample = maps.block(0, min(maps.n_maps, n_sample))
    positive = sample[sample > 0]
    if positive.size == 0:
        message = (
            "The sampled radio maps are entirely zero. Check that the maps are "
            "in linear units and that they are not empty."
        )
        warnings.warn(message, stacklevel=2)
        return message

    reference = float(np.median(positive))
    ratio = reference / utility_c
    if ratio > _SCALE_WARN_RATIO:
        message = (
            f"Radio-map values (median non-zero {reference:.3g}) are far above "
            f"utility_c = {utility_c:.3g}: W(x) is saturated almost everywhere, "
            f"so the placement is driven only by dead zones. Consider raising "
            f"utility_c towards the typical served rate."
        )
    elif ratio < 1.0 / _SCALE_WARN_RATIO:
        message = (
            f"Radio-map values (median non-zero {reference:.3g}) are far below "
            f"utility_c = {utility_c:.3g}: W(x) is effectively linear, so the "
            f"run maximises the mean field with little diminishing-returns "
            f"behaviour. Consider lowering utility_c, or check the map units "
            f"(linear, not dB)."
        )
    else:
        return None
    warnings.warn(message, stacklevel=2)
    return message


def place_towers(
    radio_maps: Union[RadioMapSet, np.ndarray, PathLike],
    locations: Union[np.ndarray, PathLike, None] = None,
    n_towers: int = 10,
    *,
    aggregation: str = "max",
    utility_c: float = 1e8,
    density: Union[np.ndarray, PathLike, None] = None,
    fixed_towers: Union[np.ndarray, PathLike, None] = None,
    allow_repeats: Optional[bool] = None,
    output: Optional[PathLike] = None,
    progress: bool = True,
    verbose: bool = True,
) -> PlacementResult:
    """Choose the best *n_towers* transmitter sites for a set of radio maps.

    This is the high-level entry point: radio maps in, tower locations out.

    Parameters
    ----------
    radio_maps : array, path or RadioMapSet
        One radio map per candidate site, as an ``(N, H, W)`` array, a path to
        a stacked ``.npy`` / ``.npz`` / text file, a directory of ``{i}.npy``
        files, or a ready-made :class:`~ia_spa.data.RadioMapSet`.  Values must
        be a non-negative field in linear units -- achievable rate in bps for
        the default *utility_c*, but any monotone quality measure works as long
        as *utility_c* is on the same scale.
    locations : array or path, optional
        Candidate site coordinates, ``(N, 3)`` as (x, y, z) or ``(N, 2)`` as
        (x, y), in the same order as the radio maps.  Optional only when
        *radio_maps* is a directory that carries its own coordinate manifest.
    n_towers : int
        How many transmitters to place.
    aggregation : {"max", "sum"}
        ``"max"`` (P_MAX) models association with the strongest base station
        and is the realistic default; ``"sum"`` (P_SUM) is the idealised upper
        bound of the paper.
    utility_c : float
        Saturation constant of W(x) = x / (x + c), in the same units as the
        radio maps.  The default 1e8 corresponds to 100 Mbps.
    density : array or path, optional
        Spatial priority weight per map cell, e.g. a population density.  Same
        shape as one radio map; normalised internally to sum to one.  Uniform
        by default.
    fixed_towers : array or path, optional
        Coordinates of transmitters that already exist.  They are snapped to
        their nearest candidate sites and held fixed, so the run returns the
        best *n_towers* additions to an existing deployment.
    allow_repeats : bool, optional
        Whether one candidate site may be chosen more than once.  Defaults to
        ``False`` under ``"max"`` (a repeat adds nothing) and ``True`` under
        ``"sum"``.
    output : path, optional
        Where to write the chosen tower locations.  A path with a suffix is
        written directly (``.csv``, ``.txt`` or ``.npy``); a path without one
        is treated as a directory receiving ``towers.csv`` and ``summary.json``.
    progress : bool
        Show a progress bar while candidates are scored.
    verbose : bool
        Print the problem summary and each selection.

    Returns
    -------
    PlacementResult
        ``result.positions`` holds the ``(n_towers, 3)`` tower coordinates;
        see :class:`~ia_spa.optimizer.PlacementResult` for the gains, the S(T)
        trajectory and the chosen candidate indices.

    Examples
    --------
    Files in, file out::

        place_towers("maps.npy", "sites.csv", n_towers=15, output="towers.csv")

    Arrays in, array out::

        result = place_towers(maps, sites, n_towers=15, verbose=False)
        result.positions
    """
    maps = _as_map_set(radio_maps, locations)

    if verbose:
        print(maps.describe())
    check_utility_scale(maps, utility_c)

    optimizer = TowerOptimizer(
        maps,
        aggregation=aggregation,
        utility_c=utility_c,
        density=_as_array(density, load_radio_maps),
    )

    result = greedy_select(
        optimizer,
        n_towers=n_towers,
        fixed_towers=_as_array(fixed_towers, load_locations),
        allow_repeats=allow_repeats,
        progress=progress,
        verbose=verbose,
    )

    if output is not None:
        written = result.save(output)
        if verbose:
            print(f"Tower locations -> {written}")

    return result


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------

def _as_map_set(
    radio_maps: Union[RadioMapSet, np.ndarray, PathLike],
    locations: Union[np.ndarray, PathLike, None],
) -> RadioMapSet:
    """Turn the (radio_maps, locations) argument pair into a RadioMapSet."""
    if isinstance(radio_maps, RadioMapSet):
        if locations is not None:
            raise ValueError(
                "'locations' must not be given when radio_maps is already a "
                "RadioMapSet -- it carries its own coordinates."
            )
        return radio_maps

    if isinstance(radio_maps, (str, Path)):
        return RadioMapSet.from_files(radio_maps, locations)

    coords = coerce_locations(locations)
    if coords is None:
        raise ValueError(
            "'locations' is required when radio maps are passed as an array."
        )
    return RadioMapSet(maps=np.asarray(radio_maps), locations=coords)


def _as_array(value, loader):
    """Load *value* with *loader* if it is a path; pass arrays through."""
    if value is None or isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, Path)):
        return np.asarray(loader(value))
    return np.asarray(value)
