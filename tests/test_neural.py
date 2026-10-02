"""Small synthetic tests for raw-sequence isolation and reusable CNN inference."""
import json

import h5py
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ncmapss_rul.features import extract_features
from ncmapss_rul.modeling import OBSERVABLE_VARIABLES
from ncmapss_rul.neural import (
    SmallRULCNN, cache_sequences, fit_channel_scaler, predict_cnn, run_cnn,
)


@pytest.fixture
def tiny_source(tmp_path):
    path = tmp_path / "explicitly_synthetic.h5"
    with h5py.File(path, "w") as handle:
        handle["W_var"] = np.array(OBSERVABLE_VARIABLES[:4], dtype="S")
        handle["X_s_var"] = np.array(OBSERVABLE_VARIABLES[4:], dtype="S")
        handle["A_var"] = np.array(["Fc", "cycle", "unit", "hs"], dtype="S")
        for split, shift in (("dev", 0), ("test", 1000)):
            counts = [(2, 1, 9), (2, 2, 6), (5, 1, 8)]
            arrays, metadata, labels = [], [], []
            for unit, cycle, count in counts:
                arrays.append((np.arange(count)[:, None] + np.arange(18)[None, :] * 10 + cycle * 100 + shift).astype(float))
                metadata.extend([[3, cycle, unit, 1]] * count)
                labels.extend([float(10 - cycle)] * count)
            values = np.concatenate(arrays)
            handle[f"W_{split}"] = values[:, :4]
            handle[f"X_s_{split}"] = values[:, 4:]
            handle[f"A_{split}"] = np.array(metadata)
            handle[f"Y_{split}"] = np.array(labels)[:, None]
            handle[f"X_v_{split}"] = np.full((len(values), 14), 999999.)
            handle[f"T_{split}"] = np.full((len(values), 10), 999999.)
    return path


def test_sequences_match_exact_windows_and_reset_at_flights(tiny_source, tmp_path):
    frame = extract_features(tiny_source, window=4, stride=4)
    # Nonconsecutive DataFrame labels and shuffled row order cannot corrupt alignment.
    frame = frame.iloc[::-1].copy()
    frame.index = np.arange(len(frame)) * 10 + 3
    values, manifest = cache_sequences(tiny_source, frame, tmp_path / "cache", window=4, stride=4, chunk_rows=8)
    assert values.shape == (5, 18, 4)
    assert values.dtype == np.float32
    with h5py.File(tiny_source, "r") as handle:
        for index, end in enumerate(frame.row_end):
            expected = np.concatenate((handle["W_dev"][end - 3:end + 1], handle["X_s_dev"][end - 3:end + 1]), axis=1).T
            np.testing.assert_array_equal(values[index], expected)
    assert manifest["channels"] == list(OBSERVABLE_VARIABLES)
    again, checked = cache_sequences(tiny_source, frame, tmp_path / "cache", window=4, stride=4)
    np.testing.assert_array_equal(values, again)
    assert checked["cache_sha256"] == manifest["cache_sha256"]


def test_sequence_reads_bounded_and_privileged_arrays_never_read(tiny_source, tmp_path, monkeypatch):
    frame = extract_features(tiny_source, window=4, stride=4)
    original = h5py.Dataset.__getitem__
    seen = []
    def checked(dataset, selection):
        name = dataset.name.lstrip("/")
        assert not name.startswith(("T_", "X_v_"))
        if name.endswith("_dev"):
            assert isinstance(selection, slice)
            assert selection.stop - selection.start <= 8
            seen.append(name)
        return original(dataset, selection)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked)
    cache_sequences(tiny_source, frame, tmp_path / "cache", window=4, stride=4, chunk_rows=8)
    assert set(seen) == {"W_dev", "X_s_dev", "Y_dev", "A_dev"}


def test_target_free_sequences_do_not_need_or_read_y(tiny_source, tmp_path, monkeypatch):
    frame = extract_features(tiny_source, window=4, stride=4).drop(columns="target")
    with h5py.File(tiny_source, "r+") as handle:
        del handle["Y_dev"]
    original = h5py.Dataset.__getitem__
    def checked(dataset, selection):
        assert not dataset.name.startswith("/Y_")
        return original(dataset, selection)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked)
    values, manifest = cache_sequences(tiny_source, frame, tmp_path / "inference", window=4, stride=4)
    assert values.shape == (5, 18, 4)
    assert manifest["target_verified"] is False


@pytest.mark.parametrize("change", ["cross_flight", "unaligned", "wrong_target", "wrong_unit"])
def test_invalid_source_keys_rejected(tiny_source, tmp_path, change):
    frame = extract_features(tiny_source, window=4, stride=4)
    if change == "cross_flight":
        frame.loc[0, "row_end"] = 10
    elif change == "unaligned":
        frame.loc[0, "row_end"] = 4
    elif change == "wrong_target":
        frame.loc[0, "target"] += 1
    else:
        frame.loc[0, "unit"] = 999
    with pytest.raises(ValueError):
        cache_sequences(tiny_source, frame, tmp_path / "cache", window=4, stride=4)


@pytest.mark.parametrize("change", ["source", "keys", "window", "cache_bytes", "channels"])
def test_stale_or_corrupted_sequence_cache_rejected(tiny_source, tmp_path, change):
    frame = extract_features(tiny_source, window=4, stride=4)
    cache = tmp_path / "cache"
    values, _ = cache_sequences(tiny_source, frame, cache, window=4, stride=4)
    del values
    window = 4
    if change == "source":
        with h5py.File(tiny_source, "r+") as handle:
            handle["W_dev"][0, 0] += 1
    elif change == "keys":
        frame = frame.iloc[::-1]
    elif change == "window":
        window = 2
    elif change == "cache_bytes":
        with (cache / "dev.f32").open("r+b") as handle:
            handle.write(np.array([9999.], dtype=np.float32).tobytes())
    else:
        manifest = json.loads((cache / "dev.json").read_text())
        manifest["channels"][0] = "forbidden"
        (cache / "dev.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cache"):
        cache_sequences(tiny_source, frame, cache, window=window, stride=window)


def test_channel_scaler_uses_only_training_partition_with_correct_weights():
    raw = np.empty((4, 18, 3), dtype=np.float32)
    raw[0] = 1
    raw[1] = 5
    raw[2:] = 1e8  # Validation/test-style extreme values cannot affect fitted stats.
    stats = fit_channel_scaler(raw, np.array([0, 1]), np.array([1., 3.]), batch_size=1)
    np.testing.assert_allclose(stats["mean"], 4.)
    np.testing.assert_allclose(stats["scale"], np.sqrt(3.))
    raw[2:] = -1e10
    assert stats == fit_channel_scaler(raw, np.array([0, 1]), np.array([1., 3.]), batch_size=1)
    constant = fit_channel_scaler(raw, np.array([0]), np.array([1.]))
    np.testing.assert_array_equal(constant["scale"], np.ones(18))


def test_saved_checkpoint_target_free_predictions_are_positive_and_identical(tmp_path):
    torch.manual_seed(410)
    model = SmallRULCNN().eval()
    raw = np.random.default_rng(410).normal(size=(7, 18, 60)).astype(np.float32)
    mean = np.arange(18) * 0.05
    scale = np.arange(18) * 0.02 + 1
    checkpoint = {"state_dict": model.state_dict(), "scaler": {"mean": mean.tolist(), "scale": scale.tolist()},
                  "channels": list(OBSERVABLE_VARIABLES), "config": {"window": 60, "target_scale": 100.0}}
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    normalized = (raw - mean.astype(np.float32)[None, :, None]) / scale.astype(np.float32)[None, :, None]
    with torch.inference_mode():
        expected = (model(torch.from_numpy(normalized)) * 100).numpy()
    actual = predict_cnn(path, raw, batch_size=3, device="cpu")
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert (actual > 0).all()
    assert actual.shape == (7,)
    with pytest.raises(ValueError, match="shape"):
        predict_cnn(path, raw.transpose(0, 2, 1), device="cpu")


def test_synthetic_full_cnn_pipeline_preserves_folds_and_reload_predictions(tmp_path, monkeypatch):
    from ncmapss_rul import neural
    from ncmapss_rul.modeling import DEV_UNITS, TEST_UNITS
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(neural, "_figures", lambda *args: {})
    source = tmp_path / "synthetic_all_engines.h5"
    with h5py.File(source, "w") as handle:
        handle["W_var"] = np.array(OBSERVABLE_VARIABLES[:4], dtype="S")
        handle["X_s_var"] = np.array(OBSERVABLE_VARIABLES[4:], dtype="S")
        handle["A_var"] = np.array(["unit", "cycle", "Fc", "hs"], dtype="S")
        for split, units in (("dev", DEV_UNITS), ("test", TEST_UNITS)):
            values, metadata, labels = [], [], []
            for unit in sorted(units):
                for cycle in (1, 2):
                    values.append(np.arange(60)[:, None] * 0.01 + np.arange(18)[None, :] + unit + cycle)
                    metadata.extend([[unit, cycle, 3, 1]] * 60)
                    labels.extend([float(2 - cycle)] * 60)
            array = np.concatenate(values)
            handle[f"W_{split}"] = array[:, :4]
            handle[f"X_s_{split}"] = array[:, 4:]
            handle[f"A_{split}"] = np.array(metadata)
            handle[f"Y_{split}"] = np.array(labels)[:, None]
    dev = extract_features(source)
    test = extract_features(source, split="test")
    output, cache = tmp_path / "output", tmp_path / "sequences"
    result = run_cnn(source, dev, test, output, cache, epochs=1, batch_size=4)
    assert result["selected_model"] == "cnn_1d"
    assert result["candidate_only"] is True
    assert result["cv_n_heldout_engines"] == 6
    assert "reused-test" in result["test_reuse_disclosure"]
    protocol = json.loads((output / "protocol.json").read_text())
    validation_seen = []
    for fold in protocol["fold_scalers"]:
        assert set(fold["train_units"]).isdisjoint(fold["validation_units"])
        assert set(fold["train_units"] + fold["validation_units"]) == DEV_UNITS
        validation_seen.extend(fold["validation_units"])
    assert set(validation_seen) == DEV_UNITS and len(validation_seen) == 6
    predictions = pd.read_parquet(output / "test_window_predictions.parquet")
    sequences, _ = cache_sequences(source, test, cache)
    reloaded = predict_cnn(output / "model.pt", sequences, device="cpu", batch_size=4)
    np.testing.assert_allclose(reloaded, predictions.prediction, rtol=1e-6)
    assert (predictions.prediction >= 0).all()
    history = pd.read_csv(output / "training_history.csv")
    assert len(history) == 4
    assert set(history.phase) == {"fold_1", "fold_2", "fold_3", "full_dev"}
