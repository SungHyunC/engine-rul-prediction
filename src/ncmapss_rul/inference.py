"""Development-only model selection and label-free HDF5 prediction."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import joblib
import numpy as np
import pandas as pd

from .features import extract_features
from .modeling import FEATURE_COLUMNS, OBSERVABLE_VARIABLES


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def choose_candidate(candidates: list[dict]) -> dict:
    """Never access test metrics while choosing the final model."""
    if not candidates or any(not np.isfinite(row["cv_rmse"]) for row in candidates):
        raise ValueError("Model candidates need finite development CV scores")
    return min(candidates, key=lambda row: (row["cv_rmse"], row["candidate"]))


def finalize_model(output: str | Path) -> dict:
    output = Path(output).resolve()
    baseline = _read(output / "results/metrics.json")
    sensitivity = _read(output / "extended/sensitivity/metrics.json")
    neural = _read(output / "extended/cnn/metrics.json")
    name = baseline["selected_model"]
    base_score = next(row["macro_engine_cycle_rmse"] for row in baseline["cv_summary"] if row["model"] == name)
    candidates = [
        {"candidate": f"original_{name}", "kind": "sklearn", "cv_rmse": base_score,
         "model_path": str(output / "results" / f"model_{name}.joblib"),
         "config": {"window": 60, "stride": 60, "model": name, "nonnegative": False},
         "test_metrics": baseline["test_metrics"][name]},
        {"candidate": "sensitivity_winner", "kind": "sklearn", "cv_rmse": sensitivity["cv_macro_engine_cycle_rmse"],
         "model_path": str(output / "extended/sensitivity/selected_model.joblib"),
         "config": sensitivity["selected_config"], "test_metrics": sensitivity["test_metrics"]},
        {"candidate": "cnn_1d", "kind": "torch", "cv_rmse": neural["cv_macro_engine_cycle_rmse"],
         "model_path": str(output / "extended/cnn/model.pt"),
         "config": {"window": 60, "stride": 60, "model": "cnn_1d", "nonnegative": True},
         "test_metrics": neural["test_metrics"]},
    ]
    selected = choose_candidate(candidates)
    directory = output / "final"
    directory.mkdir(exist_ok=True)
    model_file = directory / ("model.pt" if selected["kind"] == "torch" else "model.joblib")
    shutil.copyfile(selected["model_path"], model_file)
    card = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_candidate": selected["candidate"], "kind": selected["kind"],
        "selection_rule": "Minimum development macro engine cycle RMSE; test metrics are not selection inputs",
        "cv_macro_engine_cycle_rmse": selected["cv_rmse"], "config": selected["config"],
        "model_file": model_file.name, "model_sha256": _sha256(model_file),
        "feature_columns": list(FEATURE_COLUMNS), "observable_variables": list(OBSERVABLE_VARIABLES),
        "test_metrics": selected["test_metrics"],
        "candidates": [{k: v for k, v in row.items() if k not in {"model_path", "test_metrics"}} for row in candidates],
        "evaluation_disclosure": "Official test engines were already inspected. Extension test scores are exploratory reuse, not a new untouched holdout. CV is used for selection and is not nested-CV evidence.",
        "prediction_scope": "Research estimates at recorded flight end. Negative outputs depend on the selected policy; not validated for aircraft maintenance.",
    }
    _write(directory / "model_card.json", card)
    print(f"Final model by development CV: {selected['candidate']} ({selected['cv_rmse']:.4f} cycles)", flush=True)
    return card


def observable_sequence_batches(dataset: Path, split: str, frame: pd.DataFrame,
                                window: int, batch_size: int = 512,
                                max_read_rows: int = 120000):
    """Read only W/X_s, with a bounded span even if invalid windows left gaps."""
    endpoints = frame.row_end.to_numpy(dtype=np.int64)
    if len(endpoints) and (endpoints.min() < window - 1 or np.any(np.diff(endpoints) <= 0)):
        raise ValueError("Prediction window endpoints must be increasing and valid")
    with h5py.File(dataset, "r") as handle:
        offset = 0
        while offset < len(endpoints):
            begin = int(endpoints[offset]) - window + 1
            stop = min(offset + batch_size, len(endpoints))
            allowed = np.searchsorted(endpoints, begin + max(window, max_read_rows), side="left")
            stop = max(offset + 1, min(stop, int(allowed)))
            end = int(endpoints[stop - 1]) + 1
            values = np.concatenate([handle[f"W_{split}"][begin:end], handle[f"X_s_{split}"][begin:end]], axis=1)
            indices = endpoints[offset:stop, None] - window + 1 - begin + np.arange(window)[None, :]
            yield offset, np.asarray(values[indices].transpose(0, 2, 1), dtype=np.float32)
            offset = stop


def predict_hdf5(dataset: str | Path, model_card: str | Path, output_dir: str | Path,
                 split: str = "test", workers: int = 4, chunk_rows: int = 120000) -> dict:
    """Predict a supported HDF5 schema without requiring, reading or returning Y."""
    dataset, model_card, output_dir = Path(dataset).resolve(), Path(model_card).resolve(), Path(output_dir).resolve()
    card = _read(model_card)
    model_path = (model_card.parent / card["model_file"]).resolve()
    if model_path.parent != model_card.parent:
        raise ValueError("Model file must be local to its model card")
    if _sha256(model_path) != card["model_sha256"]:
        raise ValueError("Saved model differs from its model card")
    config = card["config"]
    frame = extract_features(dataset, split=split, window=int(config["window"]),
                             stride=int(config["stride"]), workers=workers,
                             chunk_rows=chunk_rows, include_target=False)
    if frame.empty:
        raise ValueError("No complete finite windows to predict")
    features = card["feature_columns"]
    actual_features = [name for name in frame.columns if "__" in name]
    if actual_features != features:
        raise ValueError("Observable input schema does not match the saved model")
    output_dir.mkdir(parents=True, exist_ok=True)
    if card["kind"] == "sklearn":
        model = joblib.load(model_path)
        raw = np.asarray(model.predict(frame[features]), dtype=float)
        if raw.shape != (len(frame),) or not np.isfinite(raw).all():
            raise ValueError("Model returned malformed or nonfinite raw predictions")
        predictions = np.maximum(0, raw) if config["nonnegative"] else raw
    elif card["kind"] == "torch":
        from .neural import predict_cnn
        # A temporary memory map keeps the total sequence tensor off RAM while
        # using the same batched prediction entrypoint as training verification.
        with tempfile.TemporaryDirectory(prefix="cnn-inference-", dir=output_dir) as temporary:
            if not Path(temporary).resolve().is_relative_to(output_dir):
                raise ValueError("Temporary inference directory is outside its output directory")
            mmap_path = Path(temporary) / "sequences.npy"
            sequences = np.lib.format.open_memmap(mmap_path, mode="w+", dtype=np.float32,
                                                  shape=(len(frame), 18, int(config["window"])))
            try:
                for start, batch in observable_sequence_batches(dataset, split, frame, int(config["window"]), max_read_rows=chunk_rows):
                    sequences[start:start + len(batch)] = batch
                sequences.flush()
                predictions = np.asarray(predict_cnn(model_path, sequences, batch_size=512), dtype=float)
            finally:
                sequences._mmap.close()
                del sequences
        raw = predictions.copy()
    else:
        raise ValueError(f"Unsupported model kind: {card['kind']}")
    if predictions.shape != (len(frame),) or not np.isfinite(predictions).all():
        raise ValueError("Model returned nonfinite predictions")
    windows = frame[["split", "unit", "cycle", "row_end", "flight_class"]].copy()
    windows["raw_prediction"] = raw
    windows["prediction"] = predictions
    cycles = windows.groupby(["unit", "cycle"], as_index=False).agg(
        prediction=("prediction", "mean"), raw_prediction=("raw_prediction", "mean"),
        n_windows=("prediction", "size"))
    windows.to_parquet(output_dir / "window_predictions.parquet", index=False)
    cycles.to_csv(output_dir / "cycle_predictions.csv", index=False)
    manifest = {"dataset": str(dataset), "split": split, "model_card": str(model_card),
                "model_sha256": card["model_sha256"], "input_target_required": False,
                "n_windows": len(windows), "n_cycles": len(cycles),
                "negative_cycle_predictions": int((cycles.prediction < 0).sum()),
                "config": config, "audit": frame.attrs["audit"],
                "scope": "Cycle means are available at the end of each recorded flight; no target array was read."}
    _write(output_dir / "prediction_manifest.json", manifest)
    print(f"Predicted {len(windows):,} windows / {len(cycles)} cycles without reading labels", flush=True)
    return manifest
