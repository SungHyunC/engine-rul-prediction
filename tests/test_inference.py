"""Label-free prediction must not depend on the existence or contents of Y."""
import hashlib
import json

import h5py
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from ncmapss_rul.features import extract_features
from ncmapss_rul.inference import choose_candidate, observable_sequence_batches, predict_hdf5
from ncmapss_rul.modeling import FEATURE_COLUMNS, OBSERVABLE_VARIABLES


@pytest.fixture
def unlabeled(tmp_path):
    path = tmp_path / "unlabeled.h5"
    with h5py.File(path, "w") as h:
        h["W_var"] = np.asarray(OBSERVABLE_VARIABLES[:4], dtype="S")
        h["X_s_var"] = np.asarray(OBSERVABLE_VARIABLES[4:], dtype="S")
        h["A_var"] = np.asarray(["unit", "cycle", "Fc"], dtype="S")
        values = np.arange(20 * 18, dtype=float).reshape(20, 18)
        h["W_test"] = values[:, :4]
        h["X_s_test"] = values[:, 4:]
        h["A_test"] = np.asarray([[99, 1, 2]] * 9 + [[99, 2, 2]] * 11)
    return path


@pytest.fixture
def card(tmp_path):
    model = DummyRegressor(strategy="constant", constant=-3)
    model.fit(pd.DataFrame(np.zeros((2, 90)), columns=FEATURE_COLUMNS), np.zeros(2))
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    config = {"kind": "sklearn", "model_file": path.name,
              "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "feature_columns": list(FEATURE_COLUMNS),
              "config": {"window": 4, "stride": 3, "nonnegative": True}}
    result = tmp_path / "model_card.json"
    result.write_text(json.dumps(config), encoding="utf-8")
    return result


def test_label_free_features_match_training_features(unlabeled):
    without = extract_features(unlabeled, "test", window=4, stride=3, include_target=False)
    with h5py.File(unlabeled, "r+") as h:
        h["Y_test"] = np.zeros((20, 1))
    with_labels = extract_features(unlabeled, "test", window=4, stride=3)
    pd.testing.assert_frame_equal(without, with_labels.drop(columns="target"), check_exact=True)
    assert without.row_end.tolist() == [3, 6, 12, 15, 18]


def test_predictions_do_not_read_targets(unlabeled, card, tmp_path, monkeypatch):
    with h5py.File(unlabeled, "r+") as h:
        # Even an invalid/misaligned Y must not affect inference.
        h["Y_test"] = np.array([np.nan])
    original = h5py.Dataset.__getitem__
    def protected(dataset, key):
        if dataset.name.startswith("/Y_"):
            raise AssertionError("Inference read a target array")
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", protected)
    manifest = predict_hdf5(unlabeled, card, tmp_path / "predictions", workers=1)
    assert manifest["n_windows"] == 5 and manifest["n_cycles"] == 2
    saved = pd.read_csv(tmp_path / "predictions/cycle_predictions.csv")
    assert saved.prediction.eq(0).all() and saved.raw_prediction.eq(-3).all()
    assert "target" not in saved


def test_inference_works_without_y(unlabeled, card, tmp_path):
    assert predict_hdf5(unlabeled, card, tmp_path / "out", workers=2)["n_cycles"] == 2


def test_model_tampering_rejected(unlabeled, card, tmp_path):
    (card.parent / "model.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        predict_hdf5(unlabeled, card, tmp_path / "out", workers=1)


def test_wrong_sensor_order_rejected(unlabeled, card, tmp_path):
    with h5py.File(unlabeled, "r+") as h:
        h["W_var"][0], h["W_var"][1] = b"Mach", b"alt"
    with pytest.raises(ValueError, match="schema"):
        predict_hdf5(unlabeled, card, tmp_path / "out", workers=1)


def test_final_selection_uses_development_score_only():
    candidates = [{"candidate": "a", "cv_rmse": 4., "test_rmse": 999},
                  {"candidate": "b", "cv_rmse": 6., "test_rmse": 0}]
    assert choose_candidate(candidates)["candidate"] == "a"


def test_raw_sequence_batches_preserve_causal_window_values(unlabeled):
    frame = extract_features(unlabeled, "test", window=4, stride=3, include_target=False)
    batches = list(observable_sequence_batches(unlabeled, "test", frame, 4, batch_size=2, max_read_rows=7))
    values = np.concatenate([batch for _, batch in batches])
    raw = np.arange(20 * 18, dtype=float).reshape(20, 18)
    for index, endpoint in enumerate(frame.row_end):
        np.testing.assert_array_equal(values[index], raw[endpoint-3:endpoint+1].T)
    assert values.shape == (5, 18, 4)
