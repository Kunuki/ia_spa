"""
ia_spa/scenes.py
----------------
Resolving a scene argument to a loaded Sionna scene.

Both ``scripts/compute_basis.py`` and ``scripts/evaluate.py`` accept either a
built-in Sionna scene name or a path to a scene XML file; this module holds
that shared logic.  Sionna is imported lazily, so the rest of the package --
including the placement algorithm and its CLI -- works without it installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Short aliases for the built-in Sionna scenes.
BUILTIN_SCENES = {
    "san_francisco": "san_francisco",
    "sf": "san_francisco",
    "florence": "florence",
    "fl": "florence",
    "munich": "munich",
    "etoile": "etoile",
}


def load_scene_arg(
    scene_arg: str,
    merge_shapes: bool = True,
    frequency_hz: Optional[float] = None,
    tx_array=None,
):
    """Load a built-in scene by name, or a custom scene from an XML path.

    Parameters
    ----------
    scene_arg : str
        A key of :data:`BUILTIN_SCENES` (e.g. ``"san_francisco"``, ``"sf"``)
        or a path to a Sionna scene XML file.
    merge_shapes : bool
        Passed to ``sionna.rt.load_scene``.  Use ``True`` for ray tracing and
        ``False`` when individual scene objects need to be addressed.
    frequency_hz : float, optional
        Carrier frequency to set on the loaded scene.
    tx_array : optional
        Transmitter antenna array to set on the loaded scene.

    Returns
    -------
    The loaded Sionna scene.

    Raises
    ------
    FileNotFoundError
        If *scene_arg* is neither a known built-in name nor an existing file.
    """
    from sionna.rt import load_scene, scene as sionna_scenes  # GPU / Sionna required

    key = str(scene_arg).lower()
    if key in BUILTIN_SCENES:
        source = getattr(sionna_scenes, BUILTIN_SCENES[key])
    else:
        path = Path(scene_arg)
        if not path.exists():
            raise FileNotFoundError(
                f"'{scene_arg}' is not a recognised built-in scene name and the "
                f"file does not exist. Built-in names: {sorted(BUILTIN_SCENES)}"
            )
        source = str(path)

    scene = load_scene(source, merge_shapes=merge_shapes)
    if frequency_hz is not None:
        scene.frequency = frequency_hz
    if tx_array is not None:
        scene.tx_array = tx_array
    return scene
