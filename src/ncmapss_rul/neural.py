"""Small, fixed-protocol 1D CNN on observable, flight-contained sequences.

This is an extension after inspection of the original test results. Its test
scores are descriptive reused-test results, not a new untouched confirmation.
Only development engines determine normalization, fitting and CV comparison.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import iter_segments, validate_layout
from .modeling import (
    DEV_UNITS, TEST_UNITS, OBSERVABLE_VARIABLES, _validate_frame,
    aggregate_cycle_predictions, compute_fit_weights, evaluate_predictions,
    make_cv_splits,
)

WINDOW = 60
STRIDE = 60
TARGET_SCALE = 100.0
MODEL_NAME = "cnn_1d"
KEY_COLUMNS = ["split", "unit", "cycle", "row_end", "target"]


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key_hash(frame: pd.DataFrame) -> str:
    keys = frame[[column for column in KEY_COLUMNS if column in frame]]
    schema = [(str(name), str(dtype)) for name, dtype in keys.dtypes.items()]
    digest = hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(keys, index=False).values.tobytes())
    return digest.hexdigest()


def cache_sequences(
    dataset: str | Path, frame: pd.DataFrame, cache_dir: str | Path,
    *, window: int = WINDOW, stride: int = STRIDE, chunk_rows: int = 120_000,
) -> tuple[np.memmap, dict[str, Any]]:
    """Return a verified float32 N,C,T memmap in exactly ``frame`` row order.

    Only W/X_s are inputs; A establishes flight boundaries. If ``target`` is in
    the frame, Y checks endpoint labels; otherwise neither Y nor labels are
    needed, supporting target-free inference. HDF5 row reads are bounded by
    ``chunk_rows``. For this experiment
    windows are non-overlapping (stride == window) and start at each flight.
    Source bytes, ordered metadata keys, channel schema and cache bytes are
    hashed. An existing mismatching cache is rejected rather than silently used.
    ``window`` can be smaller in synthetic software tests; run_cnn fixes it at60.
    """
    for name, value in (("window", window), ("stride", stride), ("chunk_rows", chunk_rows)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if stride != window or chunk_rows < window:
        raise ValueError("Require stride == window and chunk_rows >= window")
    if frame.empty or not set(KEY_COLUMNS[:-1]).issubset(frame) or not frame.columns.is_unique:
        raise ValueError("A nonempty frame with unique metadata columns is required")
    splits = set(frame["split"])
    if len(splits) != 1 or not splits.issubset({"dev", "test"}):
        raise ValueError("Cache frame must contain exactly one dev or test split")
    split = next(iter(splits))
    has_target = "target" in frame
    metadata = frame[[column for column in ["unit", "cycle", "row_end", "target"] if column in frame]].to_numpy(dtype=float)
    if not np.isfinite(metadata).all() or not np.equal(metadata[:, :3], np.floor(metadata[:, :3])).all():
        raise ValueError("Cache metadata must be finite with integer window identifiers")
    if frame.duplicated(["unit", "cycle", "row_end"]).any():
        raise ValueError("Duplicate sequence keys")
    path = Path(dataset).resolve()
    source_hash = _sha256(path)
    with h5py.File(path, "r") as handle:
        layout = validate_layout(handle, split, require_target=has_target)
        if tuple(layout.variable_names) != OBSERVABLE_VARIABLES:
            raise ValueError("CNN requires the exact ordered observable W + X_s channel schema")
        segments = list(iter_segments(handle, layout, chunk_rows))
    contract = {
        "version": 1, "source_path": str(path), "source_bytes": path.stat().st_size,
        "source_sha256": source_hash, "split": split, "window": window, "stride": stride,
        "channels": list(OBSERVABLE_VARIABLES), "shape": [len(frame), 18, window],
        "dtype": "float32", "ordered_keys_sha256": _key_hash(frame), "target_verified": has_target,
    }
    directory = Path(cache_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cache_path, manifest_path = directory / f"{split}.f32", directory / f"{split}.json"
    if cache_path.exists() or manifest_path.exists():
        if not cache_path.exists() or not manifest_path.exists():
            raise ValueError("Incomplete sequence cache; remove its split files and regenerate")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in contract.items()):
            raise ValueError("Stale sequence cache: source, window, channel schema or ordered keys changed")
        expected_bytes = len(frame) * 18 * window * np.dtype("float32").itemsize
        if cache_path.stat().st_size != expected_bytes or _sha256(cache_path) != manifest.get("cache_sha256"):
            raise ValueError("Sequence cache byte hash mismatch")
        return np.memmap(cache_path, mode="r", dtype=np.float32, shape=tuple(contract["shape"])), manifest

    # Resetting the index is essential: DataFrame labels are not memmap positions.
    rows = frame.reset_index(drop=True)
    groups = {key: group for key, group in rows.groupby(["unit", "cycle"], sort=False)}
    segment_lookup = {(segment.unit, segment.cycle): segment for segment in segments}
    if not set(groups).issubset(segment_lookup):
        raise ValueError("Sequence keys contain flights absent from the source split")
    for key, group in groups.items():
        segment = segment_lookup[key]
        ends = group["row_end"].to_numpy(np.int64)
        if (np.any(ends < segment.start + window - 1) or np.any(ends >= segment.end)
                or np.any((ends - segment.start + 1 - window) % stride)):
            raise ValueError("A sequence endpoint crosses a flight boundary or violates window/stride")

    temporary = directory / f"{split}.f32.partial"
    values = np.memmap(temporary, mode="w+", dtype=np.float32, shape=tuple(contract["shape"]))
    seen = np.zeros(len(frame), dtype=bool)
    windows_per_read = chunk_rows // window
    try:
        with h5py.File(path, "r") as handle:
            for key, group in groups.items():
                segment = segment_lookup[key]
                ordered = group.sort_values("row_end")
                ends = ordered["row_end"].to_numpy(np.int64)
                indices = ordered.index.to_numpy(np.int64)
                count = (segment.end - segment.start) // window
                for offset in range(0, count, windows_per_read):
                    start = segment.start + offset * window
                    stop = segment.start + min(count, offset + windows_per_read) * window
                    selected = (ends >= start) & (ends < stop)
                    if not selected.any():
                        continue
                    block = np.concatenate((
                        np.asarray(handle[f"W_{split}"][start:stop], dtype=np.float32),
                        np.asarray(handle[f"X_s_{split}"][start:stop], dtype=np.float32),
                    ), axis=1).reshape(-1, window, 18).transpose(0, 2, 1)
                    local = (ends[selected] - start + 1) // window - 1
                    batch = block[local]
                    if not np.isfinite(batch).all():
                        raise ValueError("Sequence inputs are nonfinite")
                    if has_target:
                        targets = np.asarray(handle[f"Y_{split}"][start:stop], dtype=np.float32).reshape(-1)
                        expected = rows.iloc[indices[selected]]["target"].to_numpy(np.float32)
                        if not np.array_equal(targets[ends[selected] - start], expected):
                            raise ValueError("Source endpoint targets differ from prepared keys")
                    values[indices[selected]] = batch
                    seen[indices[selected]] = True
        if not seen.all():
            raise ValueError("Not all prepared windows were found within source flights")
        values.flush()
    except BaseException:
        del values
        temporary.unlink(missing_ok=True)
        raise
    del values
    temporary.replace(cache_path)
    manifest = {**contract, "cache_sha256": _sha256(cache_path), "cache_path": str(cache_path),
                "chunk_rows": chunk_rows, "created_at_utc": datetime.now(timezone.utc).isoformat()}
    _json(manifest_path, manifest)
    return np.memmap(cache_path, mode="r", dtype=np.float32, shape=tuple(contract["shape"])), manifest


def fit_channel_scaler(sequences: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                       batch_size: int = 1024) -> dict[str, list[float]]:
    """Weighted channel mean/std using ONLY explicitly supplied training rows.

    Each time point inherits its window's engine/cycle-balanced fit weight.
    Float64 merged central moments avoid large-offset variance cancellation.
    Constant channels have scale1. No test or validation statistics are read.
    """
    indices = np.asarray(indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(indices) == 0 or len(indices) != len(weights) or np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Nonempty training indices need matching finite positive weights")
    if sequences.ndim != 3 or sequences.shape[1] != 18 or np.any(indices < 0) or np.any(indices >= len(sequences)):
        raise ValueError("Expected valid training indices and N,18,T sequences")
    mean, moment, mass = np.zeros(18), np.zeros(18), 0.0
    for start in range(0, len(indices), batch_size):
        block = np.asarray(sequences[indices[start:start + batch_size]], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Training sequences must be finite")
        weight = weights[start:start + batch_size]
        chunk_mass = float(weight.sum() * block.shape[2])
        chunk_mean = np.sum(block * weight[:, None, None], axis=(0, 2)) / chunk_mass
        chunk_moment = np.sum((block - chunk_mean[None, :, None]) ** 2 * weight[:, None, None], axis=(0, 2))
        delta = chunk_mean - mean
        total = mass + chunk_mass
        moment += chunk_moment + delta ** 2 * (mass * chunk_mass / total)
        mean += delta * (chunk_mass / total)
        mass = total
    scale = np.sqrt(np.maximum(moment / mass, 0.0))
    scale[scale < 1e-12] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist(), "weighted_timepoint_mass": mass}


class SmallRULCNN(nn.Module):
    """18→16→32 temporal filters, global averaging and nonnegative scaled RUL."""
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(18, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 1), nn.Softplus(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


def _seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _batch(sequences: np.ndarray, indices: np.ndarray, scaler: dict, device: torch.device) -> torch.Tensor:
    values = np.asarray(sequences[indices], dtype=np.float32)
    mean = np.asarray(scaler["mean"], dtype=np.float32)[None, :, None]
    scale = np.asarray(scaler["scale"], dtype=np.float32)[None, :, None]
    return torch.from_numpy(np.ascontiguousarray((values - mean) / scale)).to(device)


def _fit_cnn(sequences: np.ndarray, frame: pd.DataFrame, indices: np.ndarray,
             *, seed: int, epochs: int, batch_size: int, device: torch.device,
             phase: str) -> tuple[SmallRULCNN, dict, list[dict]]:
    _seed(seed)
    training = frame.iloc[indices]
    weights = compute_fit_weights(training)
    scaler = fit_channel_scaler(sequences, indices, weights)
    model = SmallRULCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    labels = training["target"].to_numpy(np.float32) / TARGET_SCALE
    generator = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        started = perf_counter()
        order = generator.permutation(len(indices))
        model.train()
        squared_sum, weight_sum = 0.0, 0.0
        for start in range(0, len(order), batch_size):
            positions = order[start:start + batch_size]
            x = _batch(sequences, indices[positions], scaler, device)
            y = torch.from_numpy(labels[positions]).to(device)
            weight = torch.from_numpy(weights[positions].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            squared = (model(x) - y).square() * weight
            loss = squared.mean()
            if not torch.isfinite(loss):
                raise ValueError("CNN training produced a nonfinite loss")
            loss.backward()
            optimizer.step()
            squared_sum += float(squared.detach().sum().cpu())
            weight_sum += float(weight.sum().cpu())
        history.append({"phase": phase, "epoch": epoch, "weighted_train_mse_scaled": squared_sum / weight_sum,
                        "weighted_train_rmse_cycles": np.sqrt(squared_sum / weight_sum) * TARGET_SCALE,
                        "epoch_seconds": perf_counter() - started})
        print(f"CNN {phase} epoch {epoch:02d}/{epochs}: train weighted RMSE={history[-1]['weighted_train_rmse_cycles']:.4f} cycles", flush=True)
    return model, scaler, history


def _predict(model: SmallRULCNN, sequences: np.ndarray, indices: np.ndarray,
             scaler: dict, batch_size: int, device: torch.device) -> np.ndarray:
    output = np.empty(len(indices), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            values = _batch(sequences, indices[start:start + batch_size], scaler, device)
            output[start:start + batch_size] = (model(values) * TARGET_SCALE).cpu().numpy()
    if not np.isfinite(output).all() or np.any(output < 0):
        raise ValueError("CNN prediction must be finite and nonnegative")
    return output


def predict_cnn(checkpoint_path: str | Path, sequences: np.ndarray,
                batch_size: int = 512, device: str | torch.device | None = None) -> np.ndarray:
    """Target-free inference on raw observable float32 sequences shaped N,18,60.

    Loads the saved training-only scaler and network weights. Returns one RUL
    prediction in cycles per window. Input channel order is OBSERVABLE_VARIABLES.
    Explicitly pass device='cpu' to load a CUDA-trained artifact without a GPU.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if checkpoint["channels"] != list(OBSERVABLE_VARIABLES) or config["target_scale"] != TARGET_SCALE:
        raise ValueError("Unsupported CNN checkpoint channel or target-scale schema")
    if sequences.ndim != 3 or tuple(sequences.shape[1:]) != (18, config["window"]):
        raise ValueError("Expected raw sequences with shape N,18,60")
    model = SmallRULCNN().to(target_device)
    model.load_state_dict(checkpoint["state_dict"])
    return _predict(model, sequences, np.arange(len(sequences)), checkpoint["scaler"], batch_size, target_device)


def _figures(cycles: pd.DataFrame, history: pd.DataFrame, output: Path) -> dict[str, str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(10, 9), constrained_layout=True)
    FigureCanvasAgg(figure)
    for axis, (unit, rows) in zip(figure.subplots(3, 1), cycles.groupby("unit", sort=True)):
        rows = rows.sort_values("cycle")
        axis.plot(rows.cycle, rows.target, label="Observed RUL", color="#172b4d")
        axis.plot(rows.cycle, rows.prediction, label="CNN prediction", color="#cc5500")
        axis.set(title=f"Engine {unit} — descriptive reused-test evaluation", xlabel="Flight cycle", ylabel="RUL (cycles)")
        axis.grid(alpha=0.2)
        axis.legend()
    path = output / "test_engine_rul_curves.png"
    figure.savefig(path, dpi=160)
    training = Figure(figsize=(9, 5), constrained_layout=True)
    FigureCanvasAgg(training)
    axis = training.subplots()
    for phase, rows in history.groupby("phase", sort=False):
        axis.plot(rows.epoch, rows.weighted_train_rmse_cycles, marker=".", label=phase)
    axis.set(xlabel="Fixed epoch", ylabel="Weighted training RMSE (cycles)",
             title="CNN optimization history (training batches; no early stopping)")
    axis.grid(alpha=0.2)
    axis.legend()
    history_path = output / "training_history.png"
    training.savefig(history_path, dpi=160)
    return {"rul_curves_png": str(path), "training_history_png": str(history_path)}


def run_cnn(dataset: str | Path, dev_frame: pd.DataFrame, test_frame: pd.DataFrame,
            output_dir: str | Path, cache_dir: str | Path, seed: int = 410,
            epochs: int = 12, batch_size: int = 512) -> dict[str, Any]:
    """Fixed-epoch, 3-fold development-engine CV and descriptive reused-test fit.

    Returns the JSON-serializable metrics.json content: selected_model='cnn_1d'
    identifies this candidate (not a winner against other models); top-level
    cv_macro_engine_cycle_rmse/mae average six held-out engines, test_metrics
    follows modeling.evaluate_predictions, config records fixed hyperparameters,
    and artifacts maps file names to absolute paths. Each fold's training-only
    scaler is preserved in protocol.json; the full-development scaler is also
    in model.pt for target-free predict_cnn. Original baseline files are untouched.
    """
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    _validate_frame(dev_frame, "dev", DEV_UNITS)
    _validate_frame(test_frame, "test", TEST_UNITS)
    folds = make_cv_splits(dev_frame)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Bounded CPU work; no DataLoader child processes duplicate a Windows cache.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = {"window": WINDOW, "stride": STRIDE, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": 0.001, "target_scale": TARGET_SCALE, "seed": seed,
              "optimizer": "Adam", "loss": "engine/cycle-weighted MSE of target/100",
              "architecture": "Conv1d(18,16,5,pad2)-ReLU-Conv1d(16,32,5,pad2)-ReLU-AdaptiveAvgPool1d(1)-Linear(32,1)-Softplus"}
    protocol = {
        "config": config, "channels": list(OBSERVABLE_VARIABLES),
        "test_reuse_disclosure": "Official test outcomes were inspected in the original baseline study. Extension test metrics are descriptive reused-test evaluation, not an untouched confirmatory holdout.",
        "selection": "Three GroupKFold development-engine folds; fixed epochs, no early stopping or validation/test hyperparameter tuning. Any cross-candidate selection must use development CV only.",
        "normalization": "Channel population mean/std from current training sequences only, using mean-one engine/cycle-balanced window weights and equal time-point weight within each window.",
        "fit_weights": "1/windows_in_cycle/cycles_in_engine, normalized to mean1 within fitting partition; minibatch mean weighted squared error.",
        "prediction": "Softplus scaled by100 yields nonnegative RUL; cycle estimate averages windows at flight end. Nonnegative does not establish deployment validity.",
        "reproducibility": "Fixed Python/NumPy/PyTorch seeds; deterministic algorithms and cuDNN; exact equality across different hardware/PyTorch versions is not guaranteed.",
        "fold_scalers": [], "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Publish the protocol before training, recording the intended experiment.
    _json(output / "protocol.json", protocol)
    started_all = perf_counter()
    try:
        dev_sequences, dev_manifest = cache_sequences(dataset, dev_frame, cache_dir)
        test_sequences, test_manifest = cache_sequences(dataset, test_frame, cache_dir)
        protocol["sequence_manifests"] = {"dev": dev_manifest, "test": test_manifest}
        cv_rows, fold_rows, histories, cv_predictions = [], [], [], []
        for fold, (train_indices, validation_indices) in enumerate(folds, 1):
            train_units = sorted(dev_frame.iloc[train_indices].unit.astype(int).unique().tolist())
            validation_units = sorted(dev_frame.iloc[validation_indices].unit.astype(int).unique().tolist())
            fold_rows.append({"fold": fold, "train_units": json.dumps(train_units),
                              "validation_units": json.dumps(validation_units), "n_train_windows": len(train_indices),
                              "n_validation_windows": len(validation_indices)})
            started = perf_counter()
            model, scaler, history = _fit_cnn(dev_sequences, dev_frame, train_indices, seed=seed,
                                             epochs=epochs, batch_size=batch_size, device=device, phase=f"fold_{fold}")
            fit_seconds = perf_counter() - started
            histories.extend(history)
            protocol["fold_scalers"].append({"fold": fold, "train_units": train_units, "validation_units": validation_units,
                                             "scaler": scaler})
            predictions = dev_frame.iloc[validation_indices][KEY_COLUMNS].copy()
            predictions["prediction"] = _predict(model, dev_sequences, validation_indices, scaler, batch_size, device)
            predictions["fold"] = fold
            cv_predictions.append(predictions)
            for metric in evaluate_predictions(predictions)["cycles"]["per_engine"]:
                cv_rows.append({"model": MODEL_NAME, "fold": fold, "train_units": json.dumps(train_units),
                                "validation_units": json.dumps(validation_units), "fit_seconds": fit_seconds, **metric})
            del model
        # No test score is calculated before the fixed CV experiment is complete.
        fit_started = perf_counter()
        model, scaler, history = _fit_cnn(dev_sequences, dev_frame, np.arange(len(dev_frame)), seed=seed,
                                         epochs=epochs, batch_size=batch_size, device=device, phase="full_dev")
        fit_seconds = perf_counter() - fit_started
        histories.extend(history)
        checkpoint = {"state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                      "config": config, "scaler": scaler, "channels": list(OBSERVABLE_VARIABLES),
                      "training_units": sorted(DEV_UNITS), "source_sha256": dev_manifest["source_sha256"],
                      "test_reuse_disclosure": protocol["test_reuse_disclosure"]}
        checkpoint_path = output / "model.pt"
        torch.save(checkpoint, checkpoint_path)
        prediction_started = perf_counter()
        predictions = test_frame[[c for c in [*KEY_COLUMNS, "flight_class"] if c in test_frame]].copy()
        predictions["prediction"] = _predict(model, test_sequences, np.arange(len(test_frame)), scaler, batch_size, device)
        predictions["model"] = MODEL_NAME
        prediction_seconds = perf_counter() - prediction_started
        cycles = aggregate_cycle_predictions(predictions)
        cycles["model"] = MODEL_NAME
        cv = pd.DataFrame(cv_rows)
        history_frame = pd.DataFrame(histories)
        cv.to_csv(output / "cv_results.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output / "cv_fold_assignments.csv", index=False)
        history_frame.to_csv(output / "training_history.csv", index=False)
        pd.concat(cv_predictions, ignore_index=True).to_parquet(output / "cv_window_predictions.parquet", index=False)
        predictions.to_parquet(output / "test_window_predictions.parquet", index=False)
        cycles.to_csv(output / "test_cycle_predictions.csv", index=False)
        protocol["full_dev_scaler"] = scaler
        _json(output / "protocol.json", protocol)
        artifacts = {name: str(output / name) for name in (
            "cv_results.csv", "cv_fold_assignments.csv", "training_history.csv", "model.pt",
            "cv_window_predictions.parquet", "test_window_predictions.parquet", "test_cycle_predictions.csv", "protocol.json", "metrics.json")}
        artifacts.update(_figures(cycles, history_frame, output))
        result = {"selected_model": MODEL_NAME, "candidate_only": True, "config": config,
                  "cv_macro_engine_cycle_rmse": float(cv.rmse.mean()), "cv_macro_engine_cycle_mae": float(cv.mae.mean()),
                  "cv_n_heldout_engines": int(cv.unit.nunique()), "test_metrics": evaluate_predictions(predictions),
                  "test_reuse_disclosure": protocol["test_reuse_disclosure"],
                  "runtime": {"device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                              "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                              "fit_seconds_full_dev": fit_seconds, "predict_seconds_test": prediction_seconds,
                              "total_seconds": perf_counter() - started_all, "parameter_count": sum(p.numel() for p in model.parameters()),
                              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                  "artifacts": artifacts}
        _json(output / "metrics.json", result)
        return result
    finally:
        torch.set_num_threads(old_threads)
