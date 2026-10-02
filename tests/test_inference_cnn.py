"""Tiny CPU CNN checkpoints verify label-free deployment integration on Windows."""
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ncmapss_rul.inference import predict_hdf5
from ncmapss_rul.modeling import FEATURE_COLUMNS, OBSERVABLE_VARIABLES
from ncmapss_rul.neural import SmallRULCNN, predict_cnn


@pytest.fixture
def unlabelled_cnn(tmp_path, monkeypatch):
    # No training or GPU activity is needed to check the serialized inference path.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(410)
    source = tmp_path / "unlabelled.h5"
    counts = [125, 122, 63]
    raw = np.arange(sum(counts) * 18, dtype=float).reshape(-1, 18) * 0.001
    with h5py.File(source, "w") as handle:
        handle["W_var"] = np.asarray(OBSERVABLE_VARIABLES[:4], dtype="S")
        handle["X_s_var"] = np.asarray(OBSERVABLE_VARIABLES[4:], dtype="S")
        handle["A_var"] = np.asarray(["unit", "cycle", "Fc"], dtype="S")
        handle["W_test"], handle["X_s_test"] = raw[:, :4], raw[:, 4:]
        handle["A_test"] = np.asarray([[99, cycle, 1] for cycle, count in enumerate(counts, 1) for _ in range(count)])
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": SmallRULCNN().state_dict(), "channels": list(OBSERVABLE_VARIABLES),
                "scaler": {"mean": [0.] * 18, "scale": [10.] * 18},
                "config": {"window": 60, "target_scale": 100.0}}, checkpoint)
    card = tmp_path / "model_card.json"
    card.write_text(json.dumps({"kind": "torch", "model_file": checkpoint.name,
                               "model_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                               "feature_columns": list(FEATURE_COLUMNS),
                               "config": {"window": 60, "stride": 60, "nonnegative": True}}))
    yield source, raw, checkpoint, card
    torch.set_num_threads(previous_threads)


def test_cnn_predicts_unlabelled_flights_with_exact_endpoint_alignment(unlabelled_cnn, tmp_path, monkeypatch):
    source, raw, checkpoint, card = unlabelled_cnn
    expected_endpoints = [59, 119, 184, 244, 306]
    direct_sequences = np.stack([raw[end - 59:end + 1].T for end in expected_endpoints]).astype(np.float32)
    expected = predict_cnn(checkpoint, direct_sequences, device="cpu")
    original = h5py.Dataset.__getitem__
    def guarded(dataset, selection):
        assert not dataset.name.startswith(("/Y_", "/T_", "/X_v_"))
        if dataset.name.endswith("_test"):
            assert isinstance(selection, slice)
            assert selection.stop - selection.start <= 73
        return original(dataset, selection)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    output = tmp_path / "predictions"
    manifest = predict_hdf5(source, card, output, workers=1, chunk_rows=73)
    assert manifest["input_target_required"] is False
    assert manifest["n_windows"] == 5 and manifest["n_cycles"] == 3
    windows = pd.read_parquet(output / "window_predictions.parquet")
    assert windows.row_end.tolist() == expected_endpoints
    assert windows.cycle.tolist() == [1, 1, 2, 2, 3]
    np.testing.assert_allclose(windows.prediction, expected, rtol=1e-6)
    assert "target" not in windows and (windows.prediction >= 0).all()
    assert not list(output.glob("cnn-inference-*"))


def test_cnn_does_not_read_even_malformed_y(unlabelled_cnn, tmp_path, monkeypatch):
    source, _, _, card = unlabelled_cnn
    with h5py.File(source, "r+") as handle:
        handle["Y_test"] = np.array([np.nan])
    original = h5py.Dataset.__getitem__
    def guarded(dataset, selection):
        if dataset.name.startswith("/Y_"):
            raise AssertionError("CNN inference accessed labels")
        return original(dataset, selection)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    assert predict_hdf5(source, card, tmp_path / "predictions", workers=1)["n_windows"] == 5


def test_cnn_temporary_memmap_closes_when_predictor_raises(unlabelled_cnn, tmp_path, monkeypatch):
    from ncmapss_rul import neural
    source, _, _, card = unlabelled_cnn
    def fail(*args, **kwargs):
        raise RuntimeError("sentinel inference failure")
    monkeypatch.setattr(neural, "predict_cnn", fail)
    output = tmp_path / "predictions"
    with pytest.raises(RuntimeError, match="sentinel inference failure"):
        predict_hdf5(source, card, output, workers=1)
    assert not list(output.glob("cnn-inference-*"))
    assert not (output / "window_predictions.parquet").exists()


def test_nonfinite_cnn_predictions_rejected_before_artifacts(unlabelled_cnn, tmp_path, monkeypatch):
    from ncmapss_rul import neural
    source, _, _, card = unlabelled_cnn
    monkeypatch.setattr(neural, "predict_cnn", lambda checkpoint, sequences, **kwargs: np.full(len(sequences), np.nan))
    output = tmp_path / "predictions"
    with pytest.raises(ValueError, match="nonfinite"):
        predict_hdf5(source, card, output, workers=1)
    assert not list(output.glob("cnn-inference-*"))
    assert not (output / "window_predictions.parquet").exists()


def test_cnn_checkpoint_hash_checked_before_loading(unlabelled_cnn, tmp_path):
    source, _, checkpoint, card = unlabelled_cnn
    with checkpoint.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="differs"):
        predict_hdf5(source, card, tmp_path / "predictions", workers=1)


@pytest.mark.parametrize("raw_value", [-np.inf, np.inf, np.nan])
def test_clipping_cannot_hide_nonfinite_sklearn_raw_outputs(unlabelled_cnn, tmp_path, monkeypatch, raw_value):
    from types import SimpleNamespace
    from ncmapss_rul import inference
    source, _, _, card = unlabelled_cnn
    payload = json.loads(card.read_text())
    payload["kind"] = "sklearn"
    card.write_text(json.dumps(payload))
    # Stub only deserialization to isolate raw-output numeric validation.
    monkeypatch.setattr(inference.joblib, "load", lambda path: SimpleNamespace(
        predict=lambda frame: np.full(len(frame), raw_value)))
    output = tmp_path / "predictions"
    with pytest.raises(ValueError, match="finite"):
        predict_hdf5(source, card, output, workers=1)
    assert not (output / "window_predictions.parquet").exists()
