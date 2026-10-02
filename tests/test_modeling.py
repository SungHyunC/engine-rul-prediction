"""Model tests use explicitly synthetic data, never reported project results."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from ncmapss_rul.modeling import (
    DEV_UNITS,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    TEST_UNITS,
    aggregate_cycle_predictions,
    compute_fit_weights,
    evaluate_predictions,
    get_feature_columns,
    make_cv_splits,
    train_evaluate,
)


def _synthetic_features(units, split, seed=410):
    """Synthetic values used exclusively to verify software behavior."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit in sorted(units):
        for cycle in range(1, 7):
            for window in range(1 + cycle % 3):
                row = {
                    "split": split, "unit": unit, "cycle": cycle,
                    "row_end": cycle * 1000 + window * 100,
                    "flight_class": 1 + unit % 3, "target": float(6 - cycle),
                }
                for index, feature in enumerate(FEATURE_COLUMNS):
                    row[feature] = float(cycle + unit * 0.01 + index * 0.001 + rng.normal(0, 0.05))
                rows.append(row)
    return pd.DataFrame(rows)


def test_grouped_folds_never_mix_engines_or_use_test_engines():
    dev = _synthetic_features(DEV_UNITS, "dev")
    seen = []
    for training_indices, validation_indices in make_cv_splits(dev):
        training_units = set(dev.iloc[training_indices]["unit"])
        validation_units = set(dev.iloc[validation_indices]["unit"])
        assert training_units.isdisjoint(validation_units)
        assert (training_units | validation_units) == DEV_UNITS
        assert (training_units | validation_units).isdisjoint(TEST_UNITS)
        seen.extend(validation_units)
    assert sorted(seen) == sorted(DEV_UNITS)


def test_fit_weights_balance_engines_and_cycles_despite_window_counts():
    frame = pd.DataFrame({"unit": [2, 2, 2, 5], "cycle": [1, 1, 2, 1]})
    frame["weight"] = compute_fit_weights(frame)
    assert frame["weight"].tolist() == [0.5, 0.5, 1.0, 2.0]
    np.testing.assert_allclose(frame.groupby("unit")["weight"].sum(), [2, 2])
    assert frame["weight"].mean() == 1.0
    assert frame["weight"].sum() == len(frame)


def test_cycle_metrics_average_predictions_then_weight_engines_equally():
    predictions = pd.DataFrame({
        "unit": [2, 2, 5, 5, 5], "cycle": [1, 1, 1, 2, 3],
        "target": [2, 2, 2, 1, 0], "prediction": [0, 4, 12, 11, 10],
    })
    cycles = aggregate_cycle_predictions(predictions)
    assert cycles.loc[cycles["unit"] == 2, "prediction"].item() == 2.0
    metrics = evaluate_predictions(predictions)
    assert metrics["cycles"]["macro_engine_rmse"] == 5.0
    assert metrics["cycles"]["macro_engine_mae"] == 5.0
    assert metrics["cycles"]["micro_rmse"] == pytest.approx(np.sqrt(75))
    assert metrics["windows"]["macro_engine_rmse"] == 6.0
    assert metrics["cycles"]["n_cycles"] == 4
    assert metrics["windows"]["n_windows"] == 5


@pytest.mark.parametrize("extra", ["hs", "T", "X_v", "X_v__mean", "target__mean"])
def test_unknown_or_privileged_features_fail_schema(extra):
    dev = _synthetic_features(DEV_UNITS, "dev")
    assert get_feature_columns(dev) == list(FEATURE_COLUMNS)
    assert not set(["unit", "cycle", "row_end", "flight_class", "target"]) & set(get_feature_columns(dev))
    dev[extra] = 1.0
    with pytest.raises(ValueError, match="schema mismatch"):
        get_feature_columns(dev)


@pytest.mark.parametrize("fault", ["split", "engine", "missing_feature", "nan", "duplicate", "cycle_target"])
def test_invalid_inputs_fail_before_writing_outputs(tmp_path, fault):
    dev = _synthetic_features(DEV_UNITS, "dev")
    test = _synthetic_features(TEST_UNITS, "test")
    if fault == "split":
        test.loc[0, "split"] = "dev"
    elif fault == "engine":
        test.loc[test["unit"] == 11, "unit"] = 2
    elif fault == "missing_feature":
        test = test.drop(columns=FEATURE_COLUMNS[0])
    elif fault == "nan":
        test.loc[0, FEATURE_COLUMNS[0]] = np.nan
    elif fault == "duplicate":
        test = pd.concat([test, test.iloc[[0]]], ignore_index=True)
    elif fault == "cycle_target":
        test.loc[0, "target"] += 1
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError):
        train_evaluate(dev, test, output)
    assert not output.exists()


def test_complete_synthetic_run_selection_and_train_only_scaling(tmp_path):
    dev = _synthetic_features(DEV_UNITS, "dev")
    test = _synthetic_features(TEST_UNITS, "test", seed=123)
    # Large test-only distribution shift would expose fitting the scaler on test.
    test.loc[:, list(FEATURE_COLUMNS)] += 100.0
    with threadpool_limits(limits=2):
        result = train_evaluate(dev, test, tmp_path / "synthetic_outputs")
    assert result["selection_metric"] == "dev_cv_macro_engine_cycle_rmse"
    expected = min(result["cv_summary"], key=lambda row: row["macro_engine_cycle_rmse"])["model"]
    assert result["selected_model"] == expected
    assert set(result["test_metrics"]) == set(MODEL_NAMES)
    assert result["feature_columns"] == list(FEATURE_COLUMNS)
    ridge = joblib.load(result["artifacts"]["models"]["ridge"])
    expected_mean = np.average(dev[list(FEATURE_COLUMNS)], axis=0, weights=compute_fit_weights(dev))
    np.testing.assert_allclose(ridge.named_steps["scaler"].mean_, expected_mean)
    cv_results = pd.read_csv(result["artifacts"]["cv_results_csv"])
    assert set(cv_results["unit"]) == DEV_UNITS
    for name in MODEL_NAMES:
        assert len(cv_results[cv_results["model"] == name]) == len(DEV_UNITS)
        assert result["test_metrics"][name]["cycles"]["n_engines"] == 3
    predictions = pd.read_parquet(result["artifacts"]["test_window_predictions_parquet"])
    assert len(predictions) == len(test) * len(MODEL_NAMES)
    assert set(predictions["split"]) == {"test"}
    assert set(predictions["unit"]) == TEST_UNITS
    assert set(predictions.loc[predictions["selected_by_dev_cv"], "model"]) == {expected}
    saved = json.loads(Path(result["artifacts"]["metrics_json"]).read_text(encoding="utf-8"))
    assert saved == result
    for key, value in result["artifacts"].items():
        if key == "models":
            assert all(Path(path).is_file() for path in value.values())
        else:
            assert Path(value).is_file()
