"""Exploratory window/stride sensitivity with development-only model selection.

The initial test set has already been inspected. The extension declares its
candidate grid before fitting, but its final test scores are descriptive reuse,
not a fresh confirmatory holdout. The original raw baseline is never modified.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .features import extract_features
from .modeling import (
    DEV_UNITS, TEST_UNITS, _fit, _new_model, _validate_frame,
    aggregate_cycle_predictions, evaluate_predictions, get_feature_columns,
    make_cv_splits,
)

WINDOW_STRIDE_GRID = ((30, 15), (30, 30), (60, 30), (60, 60), (120, 60), (120, 120))
MODEL_NAMES = ("ridge", "hist_gradient_boosting")
CACHE_VERSION = "sensitivity-observable-features-v1"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    schema = [(str(name), str(dtype)) for name, dtype in frame.dtypes.items()]
    digest = hashlib.sha256(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode())
    digest.update(b"\x00")
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def _extraction_code_hash() -> str:
    digest = hashlib.sha256(CACHE_VERSION.encode())
    for name in ("data.py", "features.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def _load_features_cached(dataset: Path, split: str, window: int, stride: int,
                          workers: int, chunk_rows: int, cache_dir: Path,
                          source_sha256: str, extraction_sha256: str) -> tuple[pd.DataFrame, dict]:
    """Reuse only a source-, implementation-, schema- and content-verified cache.

    A corrupt or mismatched existing entry fails closed. Worker/chunk choices
    do not identify mathematical features because their invariance is tested.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source_sha256[:16]}_{extraction_sha256[:16]}_{split}_w{window}_s{stride}"
    parquet_path = cache_dir / f"{stem}.parquet"
    manifest_path = cache_dir / f"{stem}.json"
    expected = {"cache_version": CACHE_VERSION, "source_sha256": source_sha256,
                "extraction_code_sha256": extraction_sha256, "split": split,
                "window": window, "stride": stride}
    if parquet_path.exists() or manifest_path.exists():
        if not parquet_path.exists() or not manifest_path.exists():
            raise ValueError(f"Incomplete feature cache: {stem}; remove that entry and rerun.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Feature cache provenance mismatch: {stem}.")
        if manifest.get("parquet_sha256") != _file_sha256(parquet_path):
            raise ValueError(f"Feature cache file integrity mismatch: {stem}.")
        frame = pd.read_parquet(parquet_path)
        if manifest.get("frame_sha256") != _frame_fingerprint(frame):
            raise ValueError(f"Feature cache schema/content mismatch: {stem}.")
        actual_schema = [[str(name), str(dtype)] for name, dtype in frame.dtypes.items()]
        if manifest.get("schema") != actual_schema:
            raise ValueError(f"Feature cache schema mismatch: {stem}.")
        _validate_frame(frame, split, DEV_UNITS if split == "dev" else TEST_UNITS)
        return frame, {**manifest, "cache_hit": True, "manifest_path": str(manifest_path)}
    frame = extract_features(dataset, split=split, window=window, stride=stride,
                             workers=workers, chunk_rows=chunk_rows)
    _validate_frame(frame, split, DEV_UNITS if split == "dev" else TEST_UNITS)
    frame.to_parquet(parquet_path, index=False)
    manifest = {**expected, "source_path": str(dataset), "n_windows": len(frame),
                "schema": [[str(name), str(dtype)] for name, dtype in frame.dtypes.items()],
                "frame_sha256": _frame_fingerprint(frame), "parquet_sha256": _file_sha256(parquet_path),
                "audit": frame.attrs.get("audit", {}), "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(manifest_path, manifest)
    return frame, {**manifest, "cache_hit": False, "manifest_path": str(manifest_path)}


def apply_prediction_policy(raw_predictions: np.ndarray, nonnegative: bool) -> np.ndarray:
    """Apply the predeclared policy at window level, before flight averaging."""
    raw = np.asarray(raw_predictions, dtype=np.float64)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise ValueError("Predictions must be a finite one-dimensional array.")
    return np.maximum(raw, 0.0) if nonnegative else raw.copy()


def _prediction_frame(frame: pd.DataFrame, raw: np.ndarray, nonnegative: bool) -> pd.DataFrame:
    metadata = [name for name in ("split", "unit", "cycle", "row_end", "flight_class", "target") if name in frame]
    result = frame[metadata].copy()
    result["raw_prediction"] = np.asarray(raw, dtype=np.float64)
    result["prediction"] = apply_prediction_policy(raw, nonnegative)
    return result


def aggregate_with_raw(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep raw and policy-applied means separately; clipping and means differ."""
    cycles = aggregate_cycle_predictions(predictions)
    raw = predictions.groupby(["unit", "cycle"], sort=True)["raw_prediction"].mean()
    cycles = cycles.merge(raw.rename("raw_prediction"), on=["unit", "cycle"], validate="one_to_one")
    return cycles


def rank_candidates(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Rank solely by six validation-engine RMSEs; no test values are accepted."""
    keys = ["window", "stride", "model", "nonnegative"]
    required = set(keys + ["unit", "fold", "mae", "rmse"])
    if not required.issubset(cv_results):
        raise ValueError("Incomplete development CV results.")
    if any(name.startswith("test") for name in cv_results.columns):
        raise ValueError("Test metrics cannot enter development selection.")
    rows = []
    for config, group in cv_results.groupby(keys, sort=False):
        if len(group) != len(DEV_UNITS) or set(group.unit) != DEV_UNITS or group.fold.nunique() != 3:
            raise ValueError("Each candidate must validate each official development engine once across three folds.")
        if not np.isfinite(group[["mae", "rmse"]].to_numpy()).all():
            raise ValueError("CV scores must be finite.")
        rows.append({**dict(zip(keys, config)),
                     "cv_macro_engine_cycle_mae": float(group.mae.mean()),
                     "cv_macro_engine_cycle_rmse": float(group.rmse.mean()),
                     "n_heldout_engines": len(DEV_UNITS), "n_folds": 3})
    result = pd.DataFrame(rows)
    result["_model_order"] = result.model.map({name: i for i, name in enumerate(MODEL_NAMES)})
    if result["_model_order"].isna().any():
        raise ValueError("Unexpected candidate model.")
    result = result.sort_values(["cv_macro_engine_cycle_rmse", "_model_order", "nonnegative", "window", "stride"], kind="stable")
    result = result.drop(columns="_model_order").reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    result["selected_by_dev_cv"] = result["rank"] == 1
    return result


def _diagnostics(predictions: pd.DataFrame) -> dict:
    cycles = aggregate_with_raw(predictions)
    return {level: {"count": len(frame),
                    "negative_raw_predictions": int((frame.raw_prediction < 0).sum()),
                    "negative_applied_predictions": int((frame.prediction < 0).sum()),
                    "minimum_raw_prediction": float(frame.raw_prediction.min()),
                    "minimum_applied_prediction": float(frame.prediction.min())}
            for level, frame in (("windows", predictions), ("cycles", cycles))}


def _write_plots(summary: pd.DataFrame, cycles: pd.DataFrame, output: Path) -> dict:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(10, 6.3), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = np.arange(len(WINDOW_STRIDE_GRID))
    for model, color in (("ridge", "#215b84"), ("hist_gradient_boosting", "#a76030")):
        for nonnegative, linestyle in ((False, "--"), (True, "-")):
            values = [summary.loc[(summary.window == window) & (summary.stride == stride) &
                                  (summary.model == model) & (summary.nonnegative == nonnegative),
                                  "cv_macro_engine_cycle_rmse"].item()
                      for window, stride in WINDOW_STRIDE_GRID]
            label = "Ridge" if model == "ridge" else "Gradient boosting"
            axis.plot(x, values, marker="o", linestyle=linestyle, color=color,
                      label=f"{label}: {'clip at zero' if nonnegative else 'raw'}")
    axis.set_xticks(x, [f"{window}/{stride}" for window, stride in WINDOW_STRIDE_GRID])
    axis.set(xlabel="Window / stride (samples)", ylabel="Macro engine cycle RMSE",
             title="Development-only sensitivity; fixed engine folds")
    axis.grid(alpha=.2)
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2, fontsize=9)
    cv_path = output / "cv_sensitivity.png"
    figure.savefig(cv_path, dpi=160)
    units = sorted(cycles.unit.unique())
    figure = Figure(figsize=(10, 3 * len(units)), constrained_layout=True)
    FigureCanvasAgg(figure)
    for axis, unit in zip(np.atleast_1d(figure.subplots(len(units), 1)), units):
        group = cycles.loc[cycles.unit == unit].sort_values("cycle")
        axis.plot(group.cycle, group.target, label="Observed RUL", color="#172b4d")
        axis.plot(group.cycle, group.raw_prediction, label="Raw prediction", color="#a76030", linestyle="--")
        axis.plot(group.cycle, group.prediction, label="Selected policy", color="#215b84")
        axis.set(title=f"Reused test engine {int(unit)}", xlabel="Flight cycle", ylabel="RUL (cycles)")
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle("Exploratory extension: selected configuration, reused test set")
    curves_path = output / "test_engine_rul_curves.png"
    figure.savefig(curves_path, dpi=160)
    return {"cv_sensitivity_png": str(cv_path), "test_engine_rul_curves_png": str(curves_path)}


def run_sensitivity(dataset: str | Path, output_dir: str | Path, cache_dir: str | Path,
                    seed: int = 410, workers: int = 4, chunk_rows: int = 120000) -> dict:
    """Run 24 dev-CV candidates, then refit/evaluate only the frozen winner.

    Caches are split-specific. No test feature extraction/loading occurs until
    summary.csv and selected_model_metadata.json freeze the development choice.
    The clipping policy is per window before flight averaging. Saved joblib is
    a plain sklearn estimator; its feature order/policy live in the sidecar.
    """
    for name, value in (("workers", workers), ("chunk_rows", chunk_rows)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    dataset, output, cache = Path(dataset).resolve(), Path(output_dir).resolve(), Path(cache_dir).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    source_sha256 = _file_sha256(dataset)
    extraction_sha256 = _extraction_code_hash()
    protocol = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "declared_before_extension_fitting", "dataset": str(dataset),
        "source_sha256": source_sha256, "extraction_code_sha256": extraction_sha256,
        "seed": seed, "workers": workers, "chunk_rows": chunk_rows,
        "window_stride_grid": [list(pair) for pair in WINDOW_STRIDE_GRID],
        "models": {"ridge": {"alpha": 10.0, "scaler": "training-only weighted StandardScaler"},
                   "hist_gradient_boosting": {"max_iter": 100, "max_leaf_nodes": 15, "l2_regularization": 1.0, "early_stopping": False}},
        "policies": ["raw", "max(0, window_prediction), before flight averaging"],
        "selection": "Minimum mean of six validation-engine cycle RMSEs; fixed three engine-disjoint folds anchored to 60/60 development features.",
        "tie_break": "Ridge first, raw policy first, then ascending window and stride, only for exactly equal CV RMSE.",
        "fit_weights": "Reuse baseline training-only equal-engine/equal-cycle weights normalized to mean one.",
        "test_status": "Reused test set, already inspected during initial baseline. Extension and clipping candidates are exploratory; no untouched holdout claim.",
        "baseline": "Original output/results raw baseline remains unchanged.",
        "development_units": sorted(DEV_UNITS), "test_units": sorted(TEST_UNITS),
    }
    protocol_path = output / "protocol.json"
    _write_json(protocol_path, protocol)
    cache_records = []

    def load(split: str, window: int, stride: int) -> pd.DataFrame:
        frame, record = _load_features_cached(dataset, split, window, stride, workers, chunk_rows,
                                               cache, source_sha256, extraction_sha256)
        cache_records.append(record)
        return frame

    with threadpool_limits(limits=1):
        anchor = load("dev", 60, 60)
        features = get_feature_columns(anchor)
        fold_units = []
        for train_indices, valid_indices in make_cv_splits(anchor):
            fold_units.append({"train_units": sorted(anchor.iloc[train_indices].unit.unique().astype(int).tolist()),
                               "validation_units": sorted(anchor.iloc[valid_indices].unit.unique().astype(int).tolist())})
        expected_cycles = set(map(tuple, anchor[["unit", "cycle"]].drop_duplicates().to_numpy()))
        protocol["fold_assignments"] = fold_units
        _write_json(protocol_path, protocol)
        records = []
        for window, stride in WINDOW_STRIDE_GRID:
            frame = anchor if (window, stride) == (60, 60) else load("dev", window, stride)
            if set(map(tuple, frame[["unit", "cycle"]].drop_duplicates().to_numpy())) != expected_cycles:
                raise ValueError("Window settings have different cycle coverage; comparable sensitivity requires identical development cycles.")
            print(f"Sensitivity window={window}, stride={stride}: {len(frame):,} development windows", flush=True)
            for model_name in MODEL_NAMES:
                for fold_number, assignment in enumerate(fold_units, 1):
                    training = frame.loc[frame.unit.isin(assignment["train_units"])]
                    validation = frame.loc[frame.unit.isin(assignment["validation_units"])]
                    fit_start = perf_counter()
                    model = _fit(_new_model(model_name, seed), training, features)
                    fit_seconds = perf_counter() - fit_start
                    raw = model.predict(validation[features])
                    for nonnegative in (False, True):
                        prediction = _prediction_frame(validation, raw, nonnegative)
                        scores = evaluate_predictions(prediction)
                        for engine in scores["cycles"]["per_engine"]:
                            records.append({"window": window, "stride": stride, "model": model_name,
                                            "nonnegative": nonnegative, "fold": fold_number,
                                            "train_units": json.dumps(assignment["train_units"]),
                                            "validation_units": json.dumps(assignment["validation_units"]),
                                            "fit_seconds": fit_seconds, "n_development_windows": len(frame), **engine})
            pd.DataFrame(records).to_csv(output / "cv_results.csv", index=False)
            if frame is not anchor:
                del frame
        cv_results = pd.DataFrame(records)
        summary = rank_candidates(cv_results)
        summary.to_csv(output / "summary.csv", index=False)
        winner = summary.iloc[0]
        selected = {"window": int(winner.window), "stride": int(winner.stride),
                    "model": str(winner.model), "nonnegative": bool(winner.nonnegative)}
        metadata = {"selected_config": selected, "feature_columns": features,
                    "cv_macro_engine_cycle_rmse": float(winner.cv_macro_engine_cycle_rmse),
                    "selection_metric": "dev_cv_macro_engine_cycle_rmse", "source_sha256": source_sha256,
                    "extraction_code_sha256": extraction_sha256, "seed": seed,
                    "prediction_policy": "Clip each window at zero before averaging a flight" if selected["nonnegative"] else "Raw predictions",
                    "selection_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                    "test_status": protocol["test_status"], "fold_assignments": fold_units}
        metadata_path = output / "selected_model_metadata.json"
        _write_json(metadata_path, metadata)
        print(f"Selected using development only: {selected}, CV RMSE={winner.cv_macro_engine_cycle_rmse:.6f}", flush=True)
        dev = anchor if (selected["window"], selected["stride"]) == (60, 60) else load("dev", selected["window"], selected["stride"])
        fit_start = perf_counter()
        fitted = _fit(_new_model(selected["model"], seed), dev, features)
        final_fit_seconds = perf_counter() - fit_start
        model_path = output / "selected_model.joblib"
        joblib.dump(fitted, model_path)
        # The first access to test features occurs strictly after dev selection.
        test = load("test", selected["window"], selected["stride"])
        predict_start = perf_counter()
        predictions = _prediction_frame(test, fitted.predict(test[features]), selected["nonnegative"])
        predict_seconds = perf_counter() - predict_start
    predictions["model"] = selected["model"]
    predictions["nonnegative"] = selected["nonnegative"]
    predictions["selected_by_dev_cv"] = True
    cycles = aggregate_with_raw(predictions)
    cycles["model"] = selected["model"]
    cycles["nonnegative"] = selected["nonnegative"]
    window_path, cycle_path = output / "test_window_predictions.parquet", output / "test_cycle_predictions.csv"
    predictions.to_parquet(window_path, index=False)
    cycles.to_csv(cycle_path, index=False)
    raw_predictions = predictions.assign(prediction=predictions.raw_prediction)
    cache_manifest_path = output / "cache_manifest.json"
    _write_json(cache_manifest_path, {"source_sha256": source_sha256, "extraction_code_sha256": extraction_sha256, "entries": cache_records})
    artifacts = {"protocol_json": str(protocol_path), "cv_results_csv": str(output / "cv_results.csv"),
                 "summary_csv": str(output / "summary.csv"), "selected_model_joblib": str(model_path),
                 "selected_model_metadata_json": str(metadata_path), "cache_manifest_json": str(cache_manifest_path),
                 "test_window_predictions_parquet": str(window_path), "test_cycle_predictions_csv": str(cycle_path),
                 **_write_plots(summary, cycles, output), "metrics_json": str(output / "metrics.json")}
    result = {**metadata, "selected_model": selected["model"],
              "cv_macro_engine_cycle_mae": float(winner.cv_macro_engine_cycle_mae),
              "test_metrics": evaluate_predictions(predictions), "raw_test_metrics": evaluate_predictions(raw_predictions),
              "prediction_diagnostics": _diagnostics(predictions), "n_candidates": len(summary),
              "n_cv_model_fits": len(WINDOW_STRIDE_GRID) * len(MODEL_NAMES) * 3,
              "cv_summary": summary.to_dict(orient="records"), "final_fit_seconds": final_fit_seconds,
              "test_predict_seconds": predict_seconds, "elapsed_seconds": perf_counter() - started,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": artifacts}
    _write_json(output / "metrics.json", result)
    return result
