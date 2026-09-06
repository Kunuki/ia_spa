"""
ia_spa/data.py
--------------
Loading, validating and saving the arrays that IA-SPA operates on.

The optimiser needs exactly two things:

1. **Radio maps** -- one map per candidate transmitter site, giving the
   achievable rate (or received power) at every point of the service area
   when *only* that transmitter is switched on.  These are the "basis
   functions" of the paper.
2. **Locations** -- the (x, y, z) coordinate of each candidate site, in the
   same order as the radio maps.

Both may be handed over as in-memory NumPy arrays or as files on disk.

Supported radio-map files
-------------------------
``.npy``
    ``(N, H, W)`` -- N maps of H x W cells (the usual case), or
    ``(N, P)``    -- N maps already flattened to P cells each.
    Large files are memory-mapped, so only the candidates currently being
    scored are read into RAM.
``.npz``
    A single stacked array (first match of ``radio_maps`` / ``maps`` /
    ``basis`` / ``arr_0``), or one 2-D array per candidate stored under
    numeric keys (``"0"``, ``"1"``, ...), which are stacked in numeric order.
``.csv`` / ``.txt`` / ``.tsv`` / ``.dat``
    N rows x P columns of plain text; each row is one flattened map.
directory
    One ``{i}.npy`` file per candidate (``0.npy``, ``1.npy``, ...) -- the
    layout written by :func:`ia_spa.basis_functions.compute_basis_functions`.
    A ``Sionna/`` sub-directory is detected automatically, as is the
    ``0_Coordinates.txt`` location manifest.

Supported location files
------------------------
``.npy`` or plain text, shaped ``(N, 3)`` (x, y, z) or ``(N, 2)`` (x, y --
z is filled with zeros).  A leading integer index column is recognised and
dropped, so the ``0_Coordinates.txt`` files written by the Sionna pipeline
(``i, x, y, z``) can be used directly.  An optional header line is skipped.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

__all__ = [
    "RadioMapSet",
    "coerce_locations",
    "load_radio_maps",
    "load_locations",
    "save_positions",
]

PathLike = Union[str, Path]

_TEXT_SUFFIXES = {".txt", ".csv", ".tsv", ".dat"}
_STACK_KEYS = ("radio_maps", "radio_map", "maps", "basis", "basis_functions", "arr_0")
_COORD_FILENAMES = ("0_Coordinates.txt", "coordinates.txt", "locations.txt", "locations.csv")

# Chunk of radio-map data (in bytes) read at once while scoring candidates.
# Peak working memory is roughly twice this, which keeps the optimiser
# comfortable on a laptop while still amortising disk reads.
DEFAULT_CHUNK_BYTES = 64 * 1024 ** 2


# ---------------------------------------------------------------------------
# Low-level text reading
# ---------------------------------------------------------------------------

def _sniff_delimiter(path: Path) -> Optional[str]:
    """Return the delimiter of a text table, or None for whitespace."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for delim in (",", ";", "\t"):
                if delim in line:
                    return delim
            return None
    return None


def _read_text_table(path: Path) -> np.ndarray:
    """Read a numeric text table, tolerating one header line."""
    delimiter = _sniff_delimiter(path)
    try:
        return np.loadtxt(path, delimiter=delimiter, ndmin=2)
    except ValueError:
        # Most likely a header row -- retry without it.
        try:
            return np.loadtxt(path, delimiter=delimiter, skiprows=1, ndmin=2)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse '{path}' as a numeric table "
                f"(delimiter={delimiter!r}): {exc}"
            ) from exc


def _numeric_key(key: str) -> tuple:
    """Sort key that orders '2' before '10' but still handles plain names."""
    stem = key.rsplit(".", 1)[0]
    return (0, int(stem)) if stem.isdigit() else (1, key)


# ---------------------------------------------------------------------------
# Radio maps
# ---------------------------------------------------------------------------

def load_radio_maps(path: PathLike, mmap: bool = True) -> np.ndarray:
    """Load a stack of radio maps from a single file.

    Parameters
    ----------
    path : str or Path
        ``.npy``, ``.npz`` or text file.  See the module docstring for the
        accepted layouts.  Directories are handled by
        :meth:`RadioMapSet.from_files`, not here.
    mmap : bool
        Memory-map ``.npy`` files instead of reading them fully into RAM.

    Returns
    -------
    np.ndarray, shape (N, ...)
        Stack of N radio maps.  The array may be a read-only memory map.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Radio-map file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(
            f"'{path}' is a directory. Use RadioMapSet.from_files(), which "
            f"reads one '{{i}}.npy' file per candidate."
        )

    suffix = path.suffix.lower()
    if suffix == ".npy":
        maps = np.load(path, mmap_mode="r" if mmap else None)
    elif suffix == ".npz":
        maps = _load_npz_stack(path)
    elif suffix in _TEXT_SUFFIXES:
        maps = _read_text_table(path)
    else:
        raise ValueError(
            f"Unsupported radio-map format '{suffix}'. "
            f"Use .npy, .npz, or a text file ({', '.join(sorted(_TEXT_SUFFIXES))})."
        )

    maps = np.asarray(maps) if not isinstance(maps, np.memmap) else maps
    if maps.ndim < 2:
        raise ValueError(
            f"Radio maps loaded from '{path}' have shape {maps.shape}; expected "
            f"(N, H, W) or (N, P) with one map per candidate location."
        )
    return maps


def _load_npz_stack(path: Path) -> np.ndarray:
    """Extract an (N, ...) stack from a .npz archive."""
    with np.load(path) as archive:
        keys = list(archive.files)
        if not keys:
            raise ValueError(f"'{path}' contains no arrays.")

        for key in _STACK_KEYS:
            if key in keys:
                array = archive[key]
                if array.ndim >= 2:
                    return array

        if len(keys) == 1:
            return archive[keys[0]]

        ordered = sorted(keys, key=_numeric_key)
        arrays = [archive[k] for k in ordered]
        shapes = {a.shape for a in arrays}
        if len(shapes) != 1:
            raise ValueError(
                f"'{path}' holds {len(arrays)} arrays with differing shapes "
                f"{sorted(shapes)}; cannot stack them into one radio-map set."
            )
        print(
            f"Stacking {len(arrays)} arrays from '{path.name}' "
            f"in the order: {', '.join(ordered[:5])}"
            + (", ..." if len(ordered) > 5 else "")
        )
        return np.stack(arrays)


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

def load_locations(path: PathLike) -> np.ndarray:
    """Load candidate transmitter coordinates from a file.

    Parameters
    ----------
    path : str or Path
        ``.npy`` or text file holding an (N, 3) or (N, 2) table.  A leading
        integer index column (as in ``0_Coordinates.txt``) is dropped, and a
        single header line is skipped.

    Returns
    -------
    np.ndarray, shape (N, 3), dtype float
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Locations file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        coords = np.load(path)
    elif suffix == ".npz":
        coords = _load_npz_stack(path)
    else:
        coords = _read_text_table(path)

    return _normalise_locations(np.asarray(coords, dtype=float), source=str(path))


def coerce_locations(
    locations: Union[np.ndarray, PathLike, None]
) -> Optional[np.ndarray]:
    """Accept an (N, 3) array, a path to a locations file, or None."""
    if locations is None:
        return None
    if isinstance(locations, (str, Path)):
        return load_locations(locations)
    return _normalise_locations(np.asarray(locations, dtype=float))


def _normalise_locations(coords: np.ndarray, source: str = "locations") -> np.ndarray:
    """Validate and coerce a coordinate table to shape (N, 3)."""
    coords = np.atleast_2d(np.asarray(coords, dtype=float))
    if coords.ndim != 2:
        raise ValueError(
            f"{source}: expected a 2-D (N, 3) table of coordinates, "
            f"got shape {coords.shape}."
        )

    n_cols = coords.shape[1]
    if n_cols == 4:
        index_col = coords[:, 0]
        if np.array_equal(index_col, np.arange(len(coords), dtype=float)):
            coords = coords[:, 1:]
        else:
            raise ValueError(
                f"{source}: 4 columns found but the first is not a 0..N-1 index "
                f"column. Supply (x, y, z) or (i, x, y, z)."
            )
    elif n_cols == 2:
        warnings.warn(
            f"{source}: only 2 columns (x, y) found; z is set to 0.0.",
            stacklevel=3,
        )
        coords = np.column_stack([coords, np.zeros(len(coords))])
    elif n_cols != 3:
        raise ValueError(
            f"{source}: expected 2, 3 or 4 columns, got {n_cols}. "
            f"Each row must be (x, y, z), (x, y) or (i, x, y, z)."
        )

    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{source}: coordinates contain NaN or infinite values.")
    return np.ascontiguousarray(coords)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_positions(
    path: PathLike,
    positions: np.ndarray,
    indices: Optional[Sequence[int]] = None,
    gains: Optional[Sequence[float]] = None,
) -> Path:
    """Write selected tower positions to disk.

    The format follows the file extension: ``.npy`` saves a plain (N, 3)
    array, anything else writes a text table.  ``.csv`` gets a header row and
    the optional *indices* / *gains* columns; other text files stay in the
    bare ``x, y, z`` format read by :func:`ia_spa.metrics.load_greedy_positions`.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)

    if path.suffix.lower() == ".npy":
        np.save(path, positions)
        return path

    if path.suffix.lower() == ".csv":
        columns = ["x", "y", "z"]
        table = [positions[:, 0], positions[:, 1], positions[:, 2]]
        formats = ["%.6f"] * 3
        if indices is not None:
            columns.insert(0, "candidate_index")
            table.insert(0, np.asarray(indices, dtype=float))
            formats.insert(0, "%d")
        if gains is not None:
            columns.append("marginal_gain")
            table.append(np.asarray(gains, dtype=float))
            formats.append("%.8g")
        np.savetxt(
            path,
            np.column_stack(table) if table else np.empty((0, len(columns))),
            delimiter=",",
            header=",".join(columns),
            comments="",
            fmt=formats,
        )
        return path

    np.savetxt(path, positions, delimiter=", ", fmt="%.6f")
    return path


# ---------------------------------------------------------------------------
# RadioMapSet
# ---------------------------------------------------------------------------

class RadioMapSet:
    """A stack of per-candidate radio maps plus the matching coordinates.

    Maps are exposed to the optimiser as flat vectors of ``n_cells`` values
    through :meth:`block`, which reads a contiguous range of candidates.  For
    memory-mapped or on-disk-per-file inputs only the requested block is
    materialised, which keeps peak RAM bounded regardless of how many
    candidates there are.

    Parameters
    ----------
    maps : np.ndarray, shape (N, H, W) or (N, P), optional
        In-memory or memory-mapped stack.  Mutually exclusive with *files*.
    locations : np.ndarray, shape (N, 3)
        Candidate coordinates, one per map, in the same order.
    files : sequence of Path, optional
        One ``.npy`` file per candidate, in candidate order.
    validate : bool
        Run the consistency checks in :meth:`validate` on construction.
    """

    def __init__(
        self,
        maps: Optional[np.ndarray] = None,
        locations: Optional[np.ndarray] = None,
        *,
        files: Optional[Sequence[Path]] = None,
        validate: bool = True,
    ) -> None:
        if (maps is None) == (files is None):
            raise ValueError("Provide exactly one of 'maps' or 'files'.")

        self._maps = maps
        self._files = [Path(f) for f in files] if files is not None else None

        if maps is not None:
            if maps.ndim < 2:
                raise ValueError(
                    f"Radio maps have shape {maps.shape}; expected (N, H, W) or "
                    f"(N, P), with one map per candidate location."
                )
            self.n_maps = int(maps.shape[0])
            self.map_shape = tuple(int(s) for s in maps.shape[1:])
        else:
            self.n_maps = len(self._files)
            if self.n_maps == 0:
                raise ValueError("No radio-map files supplied.")
            first = np.load(self._files[0], mmap_mode="r")
            self.map_shape = tuple(int(s) for s in first.shape)

        self.n_cells = int(np.prod(self.map_shape)) if self.map_shape else 0

        if locations is None:
            raise ValueError("Candidate 'locations' are required.")
        self.locations = _normalise_locations(locations)

        if validate:
            self.validate()

    # -- constructors ---------------------------------------------------

    @classmethod
    def from_files(
        cls,
        radio_maps: PathLike,
        locations: Union[np.ndarray, PathLike, None] = None,
        *,
        mmap: bool = True,
        validate: bool = True,
    ) -> "RadioMapSet":
        """Build a set from a radio-map file or directory, plus its locations.

        *locations* may be a path or an already-loaded ``(N, 3)`` array.  When
        *radio_maps* is a directory containing ``{i}.npy`` files and
        *locations* is omitted, the coordinate manifest inside that directory
        (``0_Coordinates.txt``) is used.
        """
        path = Path(radio_maps)
        if not path.exists():
            raise FileNotFoundError(f"Radio maps not found: {path}")

        coords = coerce_locations(locations)

        if path.is_dir():
            directory, files = _discover_map_directory(path)
            if coords is None:
                coords = _find_coordinate_file(directory)
            return cls(locations=coords, files=files, validate=validate)

        if coords is None:
            raise ValueError(
                "A locations file is required when the radio maps come from a "
                "single stacked file."
            )
        return cls(
            maps=load_radio_maps(path, mmap=mmap),
            locations=coords,
            validate=validate,
        )

    # -- access ---------------------------------------------------------

    def __len__(self) -> int:
        return self.n_maps

    def map_at(self, index: int) -> np.ndarray:
        """Return radio map *index* with its original 2-D shape."""
        return self.block(index, index + 1)[0].reshape(self.map_shape)

    def block(self, start: int, stop: int) -> np.ndarray:
        """Return candidates ``[start, stop)`` as a fresh ``(B, n_cells)`` array.

        The returned array is always a writable copy, never a view onto the
        caller's data, so the optimiser can transform it in place.
        """
        stop = min(stop, self.n_maps)
        if start >= stop:
            return np.empty((0, self.n_cells), dtype=float)

        if self._maps is not None:
            # np.array (not asarray) so callers always get a fresh, writable
            # buffer they may modify in place.
            chunk = np.array(self._maps[start:stop], dtype=float)
        else:
            chunk = np.stack(
                [np.asarray(np.load(f), dtype=float) for f in self._files[start:stop]]
            )
        return chunk.reshape(stop - start, self.n_cells)

    def blocks(self, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
        """Iterate over ``(start, stop, block)`` triples covering all candidates."""
        per_map = max(self.n_cells * 8, 1)
        size = max(1, int(chunk_bytes // per_map))
        for start in range(0, self.n_maps, size):
            stop = min(start + size, self.n_maps)
            yield start, stop, self.block(start, stop)

    # -- checks ---------------------------------------------------------

    def validate(self, sample: int = 16) -> None:
        """Check shapes, alignment and value ranges; raise on anything broken.

        A sample of *sample* maps (plus the first and last) is inspected for
        NaNs, infinities and negative values, all of which would silently
        corrupt the utility ``W(x) = x / (x + c)``.
        """
        if self.n_maps != len(self.locations):
            raise ValueError(
                f"Radio-map / location mismatch: {self.n_maps} radio maps but "
                f"{len(self.locations)} locations. They must be in 1:1 order."
            )
        if self.n_maps == 0:
            raise ValueError("At least one candidate radio map is required.")
        if self.n_cells == 0:
            raise ValueError(f"Radio maps are empty (map shape {self.map_shape}).")

        if self._files is not None:
            for f in (self._files[0], self._files[-1]):
                shape = tuple(int(s) for s in np.load(f, mmap_mode="r").shape)
                if shape != self.map_shape:
                    raise ValueError(
                        f"Radio map '{f.name}' has shape {shape}, expected "
                        f"{self.map_shape}. All maps must share one grid."
                    )

        indices = sorted({0, self.n_maps - 1, *np.linspace(
            0, self.n_maps - 1, min(sample, self.n_maps), dtype=int).tolist()})
        for i in indices:
            values = self.block(i, i + 1)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Radio map {i} contains NaN or infinite values."
                )
            if np.any(values < 0):
                raise ValueError(
                    f"Radio map {i} contains negative values (min "
                    f"{values.min():.4g}). Radio maps must be a non-negative "
                    f"rate or received-power field in *linear* units -- if your "
                    f"maps are in dB or dBm, convert them first "
                    f"(linear = 10 ** (dB / 10))."
                )

    def describe(self) -> str:
        """One-paragraph summary of the loaded data, for logs and diagnostics."""
        sample = self.block(0, min(self.n_maps, 8))
        positive = sample[sample > 0]
        scale = (
            f"median non-zero value {np.median(positive):.4g}"
            if positive.size
            else "all sampled values are zero"
        )
        source = "in memory" if self._files is None else f"{self.n_maps} files on disk"
        return (
            f"{self.n_maps} candidate sites, radio maps of shape {self.map_shape} "
            f"({self.n_cells:,} cells), {source}; {scale}."
        )


def _discover_map_directory(path: Path) -> tuple[Path, list[Path]]:
    """Find the directory holding ``{i}.npy`` maps and list them in order."""
    candidates = [path, path / "Sionna"]
    for directory in candidates:
        if not directory.is_dir():
            continue
        files = sorted(
            (f for f in directory.glob("*.npy") if f.stem.isdigit()),
            key=lambda f: int(f.stem),
        )
        if files:
            indices = [int(f.stem) for f in files]
            if indices != list(range(len(indices))):
                missing = sorted(set(range(max(indices) + 1)) - set(indices))
                raise ValueError(
                    f"Radio-map files in '{directory}' are not numbered "
                    f"0..{max(indices)} without gaps; missing: {missing[:10]}"
                    + (" ..." if len(missing) > 10 else "")
                )
            return directory, files
    raise FileNotFoundError(
        f"No numbered radio-map files ('0.npy', '1.npy', ...) found in "
        f"'{path}' or '{path / 'Sionna'}'."
    )


def _find_coordinate_file(directory: Path) -> np.ndarray:
    """Load the coordinate manifest that sits next to a directory of maps."""
    for parent in (directory, directory.parent):
        for name in _COORD_FILENAMES:
            path = parent / name
            if path.exists():
                return load_locations(path)
    raise FileNotFoundError(
        f"No coordinate file ({' / '.join(_COORD_FILENAMES)}) found in "
        f"'{directory}'. Pass the locations file explicitly."
    )
