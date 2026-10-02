"""Small synthetic tests verify experiment policy and leakage/cache boundaries."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ncmapss_rul import experiments as exp
from ncmapss_rul.modeling import DEV_UNITS, TEST_UNITS, FEATURE_COLUMNS, compute_fit_weights


def synthetic_features(split, target_offset=0.0):
    rng = np.random.default_rng(410)
    rows = []
    for unit in sorted(DEV_UNITS if split == "dev" else TEST_UNITS):
        for cycle in range(1, 7):
            for window in range(2):
                row = {"split": split, "unit": unit, "cycle": cycle,
                       "row_end": unit * 10000 + cycle * 100 + window * 10,
                       "flight_class": 3, "target": float(6 - cycle + target_offset)}
                for feature in FEATURE_COLUMNS:
                    row[feature] = np.float32(cycle + rng.normal(0, .02))
                rows.append(row)
    return pd.DataFrame(rows)


def test_clipping_is_per_window_before_averaging_and_retains_raw():
    raw = np.array([-2., 4., -5.])
    unchanged = raw.copy()
    frame = pd.DataFrame({"unit": [2, 2, 5], "cycle": [1, 1, 1], "target": [1., 1., 0.],
                          "raw_prediction": raw, "prediction": exp.apply_prediction_policy(raw, True)})
    cycles = exp.aggregate_with_raw(frame)
    assert cycles.raw_prediction.tolist() == [1.0, -5.0]
    assert cycles.prediction.tolist() == [2.0, 0.0]
    np.testing.assert_array_equal(raw, unchanged)
    diagnostics = exp._diagnostics(frame)
    assert diagnostics["windows"]["negative_raw_predictions"] == 2
    assert diagnostics["cycles"]["negative_raw_predictions"] == 1
    assert diagnostics["cycles"]["negative_applied_predictions"] == 0
    with pytest.raises(ValueError, match="finite"):
        exp.apply_prediction_policy(np.array([np.nan]), True)


def candidate_records():
    rows = []
    for model in exp.MODEL_NAMES:
        for nonnegative in (False, True):
            for i, unit in enumerate(sorted(DEV_UNITS)):
                rows.append({"window": 60, "stride": 60, "model": model, "nonnegative": nonnegative,
                             "unit": unit, "fold": i // 2 + 1, "mae": 1., "rmse": 2.})
    return pd.DataFrame(rows)


def test_selection_uses_engine_macro_and_rejects_test_columns():
    records = candidate_records()
    # Exact ties prefer the simple model and retain its raw policy.
    ranked = exp.rank_candidates(records)
    assert ranked.iloc[0].model == "ridge"
    assert not bool(ranked.iloc[0].nonnegative)
    records.loc[(records.model == "hist_gradient_boosting") & records.nonnegative, "rmse"] = 1.
    assert exp.rank_candidates(records).iloc[0].model == "hist_gradient_boosting"
    with pytest.raises(ValueError, match="Test metrics"):
        exp.rank_candidates(records.assign(test_rmse=0))
    with pytest.raises(ValueError, match="each official"):
        exp.rank_candidates(records.iloc[1:])


@pytest.mark.parametrize("tamper", ["values", "dtype", "column_name", "source"])
def test_cache_reuses_verified_features_and_rejects_tampering(tmp_path, monkeypatch, tamper):
    dataset = tmp_path / "synthetic.h5"
    dataset.write_bytes(b"synthetic source; extractor is patched")
    calls = []
    def extract(*args, **kwargs):
        calls.append(kwargs["split"])
        return synthetic_features(kwargs["split"])
    monkeypatch.setattr(exp, "extract_features", extract)
    args = (dataset, "dev", 60, 60, 1, 120000, tmp_path / "cache", exp._file_sha256(dataset), exp._extraction_code_hash())
    first, manifest = exp._load_features_cached(*args)
    second, cache_hit = exp._load_features_cached(*args)
    pd.testing.assert_frame_equal(first, second)
    assert calls == ["dev"] and cache_hit["cache_hit"]
    manifest_path = Path(manifest["manifest_path"])
    saved = json.loads(manifest_path.read_text())
    if tamper == "source":
        saved["source_sha256"] = "incorrect"
    else:
        parquet = manifest_path.with_suffix(".parquet")
        frame = pd.read_parquet(parquet)
        if tamper == "values":
            frame.loc[0, FEATURE_COLUMNS[0]] += 10
        elif tamper == "dtype":
            frame[FEATURE_COLUMNS[0]] = frame[FEATURE_COLUMNS[0]].astype(np.float64)
        else:
            frame = frame.rename(columns={FEATURE_COLUMNS[0]: "X_v__mean"})
        frame.to_parquet(parquet, index=False)
        # Updating the outer file hash must not evade semantic verification.
        saved["parquet_sha256"] = exp._file_sha256(parquet)
    manifest_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="mismatch"):
        exp._load_features_cached(*args)


def test_complete_extension_freezes_selection_before_test_and_writes_public_artifacts(tmp_path, monkeypatch):
    dataset = tmp_path / "synthetic.h5"
    dataset.write_bytes(b"synthetic source; extractor is patched")
    output = tmp_path / "output"
    calls = []
    fit_calls = []
    real_fit = exp._fit
    def fitting(model, frame, features):
        assert set(frame.split) == {"dev"}
        fit_calls.append(set(frame.unit))
        return real_fit(model, frame, features)
    def extract(*args, **kwargs):
        calls.append((kwargs["split"], kwargs["window"], kwargs["stride"]))
        if kwargs["split"] == "test":
            assert (output / "summary.csv").exists()
            assert (output / "selected_model_metadata.json").exists()
            assert (output / "selected_model.joblib").exists()
        return synthetic_features(kwargs["split"])
    monkeypatch.setattr(exp, "extract_features", extract)
    monkeypatch.setattr(exp, "_fit", fitting)
    result = exp.run_sensitivity(dataset, output, tmp_path / "cache", workers=1)
    assert result["n_candidates"] == 24
    assert result["n_cv_model_fits"] == 36
    assert len(fit_calls) == 37
    assert fit_calls[-1] == DEV_UNITS
    assert all(len(units) == 4 for units in fit_calls[:-1])
    assert calls[-1][0] == "test"
    assert sum(split == "test" for split, _, _ in calls) == 1
    assert set((window, stride) for split, window, stride in calls if split == "dev") == set(exp.WINDOW_STRIDE_GRID)
    assert len(pd.read_csv(output / "cv_results.csv")) == 24 * 6
    summary = pd.read_csv(output / "summary.csv")
    assert result["cv_macro_engine_cycle_rmse"] == pytest.approx(summary.iloc[0].cv_macro_engine_cycle_rmse)
    assert result["selected_config"]["model"] == summary.iloc[0].model
    assert result["test_metrics"]["cycles"]["n_engines"] == 3
    assert "Reused test set" in result["test_status"]
    prediction = pd.read_parquet(output / "test_window_predictions.parquet")
    assert {"raw_prediction", "prediction", "nonnegative"}.issubset(prediction)
    assert set(prediction.unit) == TEST_UNITS
    selected = joblib.load(output / "selected_model.joblib")
    if result["selected_model"] == "ridge":
        dev = synthetic_features("dev")
        expected = np.average(dev[list(FEATURE_COLUMNS)], axis=0, weights=compute_fit_weights(dev))
        np.testing.assert_allclose(selected.named_steps["scaler"].mean_, expected)
    for path in result["artifacts"].values():
        assert Path(path).is_file()
    assert json.loads((output / "metrics.json").read_text()) == result
