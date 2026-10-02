"""Read N-CMAPSS schemas and engine/cycle boundaries without loading sensor arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np


@dataclass(frozen=True)
class Segment:
    """One contiguous flight; end is exclusive and offsets refer to its split."""

    start: int
    end: int
    unit: int
    cycle: int
    flight_class: int | None


@dataclass(frozen=True)
class Layout:
    split: str
    rows: int
    variable_names: tuple[str, ...]
    unit_column: int
    cycle_column: int
    class_column: int | None


def read_names(dataset: h5py.Dataset) -> list[str]:
    """Decode both flat and column-vector byte/string variable-name datasets."""
    names = []
    for item in np.asarray(dataset[()]).reshape(-1):
        name = item.decode("utf-8") if isinstance(item, bytes) else str(item)
        names.append(name.strip().rstrip("\x00"))
    return names


def validate_layout(handle: h5py.File, split: str, require_target: bool = True) -> Layout:
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")
    required = [f"W_{split}", f"X_s_{split}", f"A_{split}", "W_var", "X_s_var", "A_var"]
    if require_target:
        required.append(f"Y_{split}")
    for name in required:
        if name not in handle or not isinstance(handle[name], h5py.Dataset):
            raise ValueError(f"Missing required HDF5 dataset: {name}")
    if handle[f"W_{split}"].ndim != 2:
        raise ValueError(f"W_{split} must be a row-aligned 2D dataset")
    rows = handle[f"W_{split}"].shape[0]
    names = []
    for prefix in ("W", "X_s", "A"):
        dataset = handle[f"{prefix}_{split}"]
        if dataset.ndim != 2 or dataset.shape[0] != rows:
            raise ValueError(f"{prefix}_{split} must be a row-aligned 2D dataset")
        if not np.issubdtype(dataset.dtype, np.number):
            raise ValueError(f"{prefix}_{split} must contain numeric values")
        variable_names = read_names(handle[f"{prefix}_var"])
        if len(variable_names) != dataset.shape[1]:
            raise ValueError(f"{prefix}_var does not match {prefix}_{split} columns")
        if prefix != "A":
            names.extend(variable_names)
    if not names or len(set(names)) != len(names) or any(not n for n in names):
        raise ValueError("W/X_s variable names must be nonempty and unique")
    if require_target:
        target = handle[f"Y_{split}"]
        if target.shape not in {(rows,), (rows, 1)} or not np.issubdtype(target.dtype, np.number):
            raise ValueError(f"Y_{split} must be numeric with shape (rows,) or (rows, 1)")
    auxiliary = [name.casefold() for name in read_names(handle["A_var"])]
    if len(set(auxiliary)) != len(auxiliary):
        raise ValueError("A_var must contain unique names")
    if "unit" not in auxiliary or "cycle" not in auxiliary:
        raise ValueError("A_var must identify unit and cycle columns")
    return Layout(split, rows, tuple(names), auxiliary.index("unit"),
                  auxiliary.index("cycle"), auxiliary.index("fc") if "fc" in auxiliary else None)


def _integer_metadata(values: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError(f"{name} metadata must contain finite integer values")
    if np.any(np.abs(values) > 2**53):
        raise ValueError(f"{name} metadata is outside the supported integer range")
    return values.astype(np.int64)


def iter_segments(handle: h5py.File, layout: Layout, chunk_rows: int = 120_000) -> Iterator[Segment]:
    """Scan only bounded A slices; reject fragmented/repeated flight identifiers."""
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")
    current = None
    current_start = 0
    seen: set[tuple[int, int]] = set()
    for start in range(0, layout.rows, chunk_rows):
        block = np.asarray(handle[f"A_{layout.split}"][start:start + chunk_rows])
        units = _integer_metadata(block[:, layout.unit_column], "unit")
        cycles = _integer_metadata(block[:, layout.cycle_column], "cycle")
        classes = (_integer_metadata(block[:, layout.class_column], "Fc")
                   if layout.class_column is not None else None)
        changes = np.flatnonzero((units[1:] != units[:-1]) | (cycles[1:] != cycles[:-1])) + 1
        boundaries = np.concatenate(([0], changes, [len(block)]))
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            unit, cycle = int(units[left]), int(cycles[left])
            flight_class = int(classes[left]) if classes is not None else None
            if classes is not None and np.any(classes[left:right] != flight_class):
                raise ValueError(f"Flight class changes within unit {unit}, cycle {cycle}")
            key = (unit, cycle, flight_class)
            if current is None:
                current = key
                current_start = start + int(left)
                seen.add((unit, cycle))
            elif key[:2] == current[:2]:
                if key[2] != current[2]:
                    raise ValueError(f"Flight class changes within unit {unit}, cycle {cycle}")
            else:
                yield Segment(current_start, start + int(left), *current)
                if (unit, cycle) in seen:
                    raise ValueError(f"Noncontiguous repeated flight: unit {unit}, cycle {cycle}")
                seen.add((unit, cycle))
                current = key
                current_start = start + int(left)
    if current is not None:
        yield Segment(current_start, layout.rows, *current)


def inspect_hdf5(path: str | Path) -> dict:
    """Return JSON-serializable shapes, names and observed engine/cycle metadata.

    This reads metadata in chunks and never reads W, X_s, X_v, T or Y values.
    Engine IDs are reported from the file, not assumed to match a release.
    """
    path = Path(path).resolve()
    result = {"path": str(path), "datasets": {}, "varnames": {}, "splits": {}}
    with h5py.File(path, "r") as handle:
        for name, dataset in handle.items():
            if isinstance(dataset, h5py.Dataset):
                result["datasets"][name] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
                if name.endswith("_var"):
                    result["varnames"][name[:-4]] = read_names(dataset)
        for split in ("dev", "test"):
            if not any(f"{prefix}_{split}" in handle for prefix in ("W", "X_s", "Y", "A")):
                continue
            layout = validate_layout(handle, split)
            units = {}
            segment_count = 0
            for segment in iter_segments(handle, layout):
                segment_count += 1
                unit = units.setdefault(segment.unit, {"unit": segment.unit, "flight_classes": set(),
                                                      "cycles": [], "rows": 0})
                if segment.flight_class is not None:
                    unit["flight_classes"].add(segment.flight_class)
                unit["cycles"].append(segment.cycle)
                unit["rows"] += segment.end - segment.start
            for unit in units.values():
                unit["flight_classes"] = sorted(unit["flight_classes"])
            result["splits"][split] = {
                "rows": layout.rows, "unit_ids": sorted(units), "segments": segment_count,
                "units": [units[k] for k in sorted(units)],
                "predictor_variables": list(layout.variable_names),
            }
    return result
