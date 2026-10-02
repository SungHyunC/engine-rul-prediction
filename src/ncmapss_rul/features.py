"""Causal, bounded-read features from observable N-CMAPSS channels only.

Windows start at each unit/cycle boundary. The first complete window has
``window`` samples; subsequent windows advance by ``stride``. Final incomplete
windows are dropped. Standard deviations use ddof=0; OLS slopes are per sample.
Any complete window with a nonfinite input, endpoint target, or float32 output
is dropped. Counts and dropped incomplete samples are exposed in df.attrs.
No interpolation, imputation, normalization or label clipping happens here.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .data import Segment, iter_segments, validate_layout

STATS = ("mean", "std", "min", "max", "slope")
METADATA_COLUMNS = ("split", "unit", "cycle", "row_end", "flight_class")
TARGET_COLUMN = "target"


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Select generated statistical predictors, excluding all target/metadata."""
    return [column for column in frame.columns if "__" in column
            and column.rsplit("__", 1)[1] in STATS]


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _summarize(values: np.ndarray) -> np.ndarray:
    """Input is (windows, variables, time); accumulation remains float64."""
    length = values.shape[-1]
    centered = np.arange(length, dtype=np.float64) - (length - 1) / 2
    denominator = np.dot(centered, centered)
    means = np.mean(values, axis=-1)
    # Center y as well to avoid cancellation from a large sensor offset.
    residuals = values - means[..., None]
    slopes = ((residuals @ centered) / denominator if denominator
              else np.zeros_like(means))
    return np.stack((means, np.sqrt(np.mean(residuals * residuals, axis=-1)),
                     np.min(values, axis=-1), np.max(values, axis=-1), slopes), axis=-1)


def _extract_segment(task: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    path, split, segment, window, stride, chunk_rows, include_target = task
    assert isinstance(segment, Segment)
    feature_blocks, target_blocks, end_blocks = [], [], []
    candidate_windows = 0
    dropped = 0
    next_start = segment.start
    next_read = segment.start
    values = labels = None
    with h5py.File(path, "r") as handle:
        w, xs = handle[f"W_{split}"], handle[f"X_s_{split}"]
        y = handle[f"Y_{split}"] if include_target else None
        n_features = (w.shape[1] + xs.shape[1]) * len(STATS)
        while next_read < segment.end:
            end = min(next_read + chunk_rows, segment.end)
            new_values = np.concatenate((np.asarray(w[next_read:end], dtype=np.float64),
                                         np.asarray(xs[next_read:end], dtype=np.float64)), axis=1)
            # Internal zero placeholders preserve the same window machinery for
            # inference. They are never exposed as labels and Y is never read.
            new_labels = (np.asarray(y[next_read:end], dtype=np.float64).reshape(-1)
                          if include_target else np.zeros(end - next_read))
            values = new_values if values is None else np.concatenate((values, new_values))
            labels = new_labels if labels is None else np.concatenate((labels, new_labels))
            next_read = end
            if len(values) < window:
                continue
            count = 1 + (len(values) - window) // stride
            candidate_windows += count
            windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
            # Bounded blocks also keep stride=1 from materializing a huge tensor.
            for offset in range(0, count, 2048):
                starts = np.arange(offset, min(count, offset + 2048)) * stride
                batch = windows[starts]
                targets = labels[starts + window - 1]
                valid = np.isfinite(batch).all(axis=(1, 2)) & np.isfinite(targets)
                dropped += int((~valid).sum())
                if not valid.any():
                    continue
                with np.errstate(over="ignore", invalid="ignore"):
                    features = _summarize(batch[valid]).reshape(int(valid.sum()), n_features).astype(np.float32)
                    kept_targets = targets[valid].astype(np.float32)
                finite_output = np.isfinite(features).all(axis=1) & np.isfinite(kept_targets)
                dropped += int((~finite_output).sum())
                feature_blocks.append(features[finite_output])
                target_blocks.append(kept_targets[finite_output])
                end_blocks.append((next_start + starts[valid] + window - 1)[finite_output])
            consumed = count * stride
            next_start += consumed
            if consumed < len(values):
                # Copy only the overlap; don't keep a large old block alive.
                values, labels = values[consumed:].copy(), labels[consumed:].copy()
            else:
                values = labels = None
                next_read = max(next_read, next_start)
    audit = {"candidate_windows": candidate_windows, "dropped_nonfinite_windows": dropped,
             "incomplete_tail_samples": max(0, segment.end - next_start)}
    return (np.concatenate(feature_blocks) if feature_blocks else np.empty((0, n_features), dtype=np.float32),
            np.concatenate(target_blocks) if target_blocks else np.empty(0, dtype=np.float32),
            np.concatenate(end_blocks).astype(np.int64) if end_blocks else np.empty(0, dtype=np.int64), audit)


def extract_features(path: str | Path, split: str = "dev", window: int = 60,
                     stride: int = 60, workers: int = 1,
                     chunk_rows: int = 120_000, include_target: bool = True) -> pd.DataFrame:
    """Extract identical ordered features for any positive worker/chunk count.

    Each spawned worker opens its own read-only HDF5 handle. Call multiprocessing
    entrypoints behind ``if __name__ == '__main__'`` on Windows (the package CLI
    follows that convention). Metadata are not predictor columns. ``row_end`` is
    the zero-based endpoint within the source dev/test split. ``flight_class``
    is nullable metadata. The return frame is materialized; sensor data are not.
    With include_target=False, Y is neither required nor read, and the returned
    frame has no target column. Use that mode when predicting unlabeled data.
    """
    for name, value in (("window", window), ("stride", stride), ("workers", workers),
                        ("chunk_rows", chunk_rows)):
        _positive_integer(value, name)
    path = str(Path(path).resolve())
    with h5py.File(path, "r") as handle:
        layout = validate_layout(handle, split, require_target=include_target)
        segments = list(iter_segments(handle, layout, chunk_rows))
    columns = [f"{name}__{stat}" for name in layout.variable_names for stat in STATS]
    audit = {"input_rows": layout.rows, "segments": len(segments), "candidate_windows": 0,
             "dropped_nonfinite_windows": 0, "incomplete_tail_samples": 0, "output_rows": 0,
             "window": window, "stride": stride, "chunk_rows": chunk_rows, "workers": workers,
             "std_ddof": 0, "slope_unit": "per_sample", "target_position": "window_end",
             "nonfinite_policy": "drop_window_if_input_endpoint_target_or_output_nonfinite"}
    tasks = [(path, split, segment, window, stride, chunk_rows, include_target) for segment in segments]
    frames = []

    def collect(results):
        for segment, (features, target, row_end, segment_audit) in zip(segments, results):
            for key, value in segment_audit.items():
                audit[key] += value
            if not len(target):
                continue
            frame = pd.DataFrame(features, columns=columns)
            frame.insert(0, TARGET_COLUMN, target)
            frame.insert(0, "flight_class", pd.array([segment.flight_class] * len(target), dtype="Int64"))
            frame.insert(0, "row_end", row_end)
            frame.insert(0, "cycle", segment.cycle)
            frame.insert(0, "unit", segment.unit)
            frame.insert(0, "split", split)
            frames.append(frame)

    if workers == 1 or not tasks:
        collect(map(_extract_segment, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            collect(executor.map(_extract_segment, tasks, chunksize=1))
    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame({name: pd.Series(dtype=np.float32) for name in columns})
        result.insert(0, TARGET_COLUMN, pd.Series(dtype=np.float32))
        result.insert(0, "flight_class", pd.Series(dtype="Int64"))
        for name in ("row_end", "cycle", "unit"):
            result.insert(0, name, pd.Series(dtype=np.int64))
        result.insert(0, "split", pd.Series(dtype="object"))
    audit["output_rows"] = len(result)
    if not include_target:
        result = result.drop(columns=[TARGET_COLUMN])
        audit["target_position"] = None
        audit["nonfinite_policy"] = "drop_window_if_input_or_output_nonfinite"
    result.attrs["audit"] = audit
    return result
