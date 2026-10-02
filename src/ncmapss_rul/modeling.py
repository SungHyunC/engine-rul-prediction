"""Engine-separated RUL model selection, evaluation, and report figures.

The official development engines are the only source for fitting and model
selection. Window predictions are averaged within an engine/flight cycle for
the primary evaluation; this is an estimate available at the end of a flight,
not an online estimate available before the flight has finished.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEV_UNITS = frozenset({2, 5, 10, 16, 18, 20})
TEST_UNITS = frozenset({11, 14, 15})
OBSERVABLE_VARIABLES = (
    "alt", "Mach", "TRA", "T2", "T24", "T30", "T48", "T50",
    "P15", "P2", "P21", "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf",
)
STATISTICS = ("mean", "std", "min", "max", "slope")
FEATURE_COLUMNS = tuple(
    f"{variable}__{statistic}"
    for variable in OBSERVABLE_VARIABLES
    for statistic in STATISTICS
)
REQUIRED_METADATA = frozenset({"split", "unit", "cycle", "row_end", "target"})
OPTIONAL_METADATA = frozenset({"flight_class"})
MODEL_NAMES = ("dummy_mean", "ridge", "hist_gradient_boosting")


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return the exact observable feature whitelist, rejecting schema drift.

    Identifiers, flight class, RUL, health state, latent degradation parameters
    (T), and virtual sensors (X_v) cannot become inputs through column discovery.
    Column order in the caller's DataFrame does not affect model input order.
    """
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed.")
    required = set(FEATURE_COLUMNS) | REQUIRED_METADATA
    missing = required - set(frame.columns)
    unexpected = set(frame.columns) - required - OPTIONAL_METADATA
    if missing or unexpected:
        raise ValueError(
            f"Feature schema mismatch: missing={sorted(missing)!r}, "
            f"unexpected={sorted(unexpected, key=str)!r}."
        )
    return list(FEATURE_COLUMNS)


def _validate_frame(frame: pd.DataFrame, split: str, units: frozenset[int]) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{split} must be a nonempty DataFrame.")
    features = get_feature_columns(frame)
    if frame["split"].isna().any() or set(frame["split"]) != {split}:
        raise ValueError(f"Expected only split={split!r} rows.")
    for column in ["unit", "cycle", "row_end", "target", *features]:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"{split}.{column} must be numeric.")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{split}.{column} contains missing or nonfinite values.")
    for column in ("unit", "cycle", "row_end"):
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{split}.{column} must contain integer identifiers.")
    if set(frame["unit"].astype(int)) != units:
        raise ValueError(f"{split} must contain exactly the official engines {sorted(units)}.")
    if (frame["cycle"] <= 0).any() or (frame["row_end"] < 0).any():
        raise ValueError("Cycles must be positive and row_end must be nonnegative.")
    if (frame["target"] < 0).any():
        raise ValueError("RUL targets must be nonnegative.")
    if frame.duplicated(["unit", "cycle", "row_end"]).any():
        raise ValueError(f"{split} contains duplicate window identifiers.")
    bounds = frame.groupby(["unit", "cycle"])["target"].agg(["min", "max"])
    if ((bounds["max"] - bounds["min"]).abs() > 1e-8).any():
        raise ValueError("Windows in one engine/cycle must share the same RUL target.")


def compute_fit_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give engines equal total weight and cycles equal weight within each engine.

    Relative weight is 1 / windows_in_its_cycle / cycles_in_its_engine, then
    divided by its fitting-partition mean. The final weights have mean 1 and
    sum to the number of fitting windows, preserving a conventional weight
    scale for Ridge/HGB regularization. Every engine has total weight
    n_fitting_windows / n_fitting_engines. Only the current fitting partition
    contributes to either the relative weights or their normalization.
    """
    if frame.empty:
        raise ValueError("Cannot compute fitting weights for an empty partition.")
    windows_in_cycle = frame.groupby(["unit", "cycle"])["cycle"].transform("size")
    cycles_in_unit = frame.groupby("unit")["cycle"].transform("nunique")
    weights = (1.0 / windows_in_cycle / cycles_in_unit).to_numpy(dtype=float)
    return weights / weights.mean()


def make_cv_splits(dev: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return three deterministic, engine-disjoint development-only folds."""
    _validate_frame(dev, "dev", DEV_UNITS)
    return list(GroupKFold(n_splits=3).split(dev, groups=dev["unit"]))


def _new_model(name: str, seed: int) -> Any:
    if name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ])
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=seed,
            early_stopping=False,
        )
    raise ValueError(f"Unknown model: {name}")


def _fit(model: Any, frame: pd.DataFrame, features: list[str]) -> Any:
    weights = compute_fit_weights(frame)
    if isinstance(model, Pipeline):
        model.fit(
            frame[features], frame["target"],
            scaler__sample_weight=weights, ridge__sample_weight=weights,
        )
    else:
        model.fit(frame[features], frame["target"], sample_weight=weights)
    return model


def aggregate_cycle_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average all window predictions in a flight; preserve one true cycle RUL."""
    required = {"unit", "cycle", "target", "prediction"}
    if not required.issubset(predictions):
        raise ValueError(f"Prediction data must include {sorted(required)}.")
    if predictions.empty:
        raise ValueError("Cannot evaluate empty predictions.")
    if not np.isfinite(predictions[["target", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError("Targets and predictions must be finite.")
    bounds = predictions.groupby(["unit", "cycle"])["target"].agg(["min", "max"])
    if ((bounds["max"] - bounds["min"]).abs() > 1e-8).any():
        raise ValueError("Windows in one engine/cycle must share the same RUL target.")
    return (
        predictions.groupby(["unit", "cycle"], as_index=False, sort=True)
        .agg(target=("target", "first"), prediction=("prediction", "mean"),
             n_windows=("prediction", "size"))
    )


def _metrics_at_level(frame: pd.DataFrame, count_name: str) -> dict[str, Any]:
    per_engine = []
    for unit, group in frame.groupby("unit", sort=True):
        error = group["prediction"].to_numpy() - group["target"].to_numpy()
        per_engine.append({
            "unit": int(unit), count_name: int(len(group)),
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
        })
    errors = frame["prediction"].to_numpy() - frame["target"].to_numpy()
    return {
        "macro_engine_mae": float(np.mean([row["mae"] for row in per_engine])),
        "macro_engine_rmse": float(np.mean([row["rmse"] for row in per_engine])),
        "micro_mae": float(np.abs(errors).mean()),
        "micro_rmse": float(np.sqrt(np.square(errors).mean())),
        count_name: int(len(frame)),
        "n_engines": len(per_engine),
        "per_engine": per_engine,
    }


def evaluate_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Report raw-window and cycle-averaged errors, with equal-engine macros.

    ``cycles.macro_engine_rmse`` first computes RMSE over each engine's cycle
    predictions, then averages those engine RMSEs. It is not a pooled RMSE.
    """
    cycles = aggregate_cycle_predictions(predictions)
    return {
        "windows": _metrics_at_level(predictions, "n_windows"),
        "cycles": _metrics_at_level(cycles, "n_cycles"),
    }


def _write_figures(
    cycle_predictions: pd.DataFrame,
    test_metrics: dict[str, Any],
    selected_model: str,
    output_dir: Path,
) -> dict[str, str]:
    # FigureCanvasAgg avoids selecting or opening an interactive GUI backend.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    selected = cycle_predictions[cycle_predictions["model"] == selected_model]
    units = sorted(selected["unit"].unique())
    figure = Figure(figsize=(10, 3.0 * len(units)), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(len(units), 1))
    for axis, unit in zip(axes, units):
        rows = selected[selected["unit"] == unit].sort_values("cycle")
        axis.plot(rows["cycle"], rows["target"], label="Observed RUL", color="#172b4d", lw=2)
        axis.plot(rows["cycle"], rows["prediction"], label="Predicted RUL", color="#e76f51", lw=1.6)
        axis.set(title=f"Held-out engine {int(unit)}", xlabel="Flight cycle", ylabel="RUL (cycles)")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
    figure.suptitle(f"Test trajectories: {selected_model}\nWindow predictions averaged at flight end")
    curves_path = output_dir / "test_engine_rul_curves.png"
    figure.savefig(curves_path, dpi=160)

    figure = Figure(figsize=(10, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    positions = np.arange(len(MODEL_NAMES))
    for offset, metric, label, color in [
        (-0.18, "macro_engine_mae", "Macro engine MAE", "#457b9d"),
        (0.18, "macro_engine_rmse", "Macro engine RMSE", "#e9c46a"),
    ]:
        values = [test_metrics[name]["cycles"][metric] for name in MODEL_NAMES]
        bars = axis.bar(positions + offset, values, width=0.36, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f", padding=3)
    labels = [name + ("\n(selected by dev CV)" if name == selected_model else "") for name in MODEL_NAMES]
    axis.set_xticks(positions, labels)
    axis.set(ylabel="Error (RUL cycles)", title="Held-out test errors after flight-level averaging")
    axis.margins(y=0.2)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    comparison_path = output_dir / "test_model_error_comparison.png"
    figure.savefig(comparison_path, dpi=160)
    return {"rul_curves_png": str(curves_path), "error_comparison_png": str(comparison_path)}


def train_evaluate(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    seed: int = 410,
) -> dict[str, Any]:
    """Select by development-engine CV, refit, and evaluate official test engines.

    Inputs have the 90 named observable features and required metadata
    ``split, unit, cycle, row_end, target``; ``flight_class`` is optional and
    excluded from model inputs. All official engines must be represented.

    Return shape (also written to ``metrics.json``)::

        {
          "selected_model": str,
          "selection_metric": "dev_cv_macro_engine_cycle_rmse",
          "feature_columns": [str, ...], "seed": int,
          "development_units": [int, ...], "test_units": [int, ...],
          "evaluation_notes": {str: str},
          "cv_summary": [{"model": str, "macro_engine_cycle_mae": float,
                           "macro_engine_cycle_rmse": float, ...}, ...],
          "test_metrics": {model: {"windows": {...}, "cycles": {...},
                                   "fit_seconds": float, "predict_seconds": float}},
          "artifacts": {"models": {model: absolute_path},
                        "cv_results_csv": absolute_path,
                        "cv_fold_assignments_csv": absolute_path,
                        "test_window_predictions_parquet": absolute_path,
                        "test_window_predictions_csv": absolute_path,
                        "test_cycle_predictions_csv": absolute_path,
                        "rul_curves_png": absolute_path,
                        "error_comparison_png": absolute_path,
                        "metrics_json": absolute_path}
        }

    Both metric levels contain macro_engine_mae/rmse, micro_mae/rmse,
    n_engines, n_windows or n_cycles, and per_engine records. Prediction files
    use long format: one row per model and window/cycle, including target and
    prediction. Models saved by joblib are fitted sklearn estimators.
    """
    _validate_frame(dev, "dev", DEV_UNITS)
    _validate_frame(test, "test", TEST_UNITS)
    features = get_feature_columns(dev)
    if features != get_feature_columns(test):
        raise ValueError("Development and test feature schemas differ.")
    folds = make_cv_splits(dev)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_assignments = []
    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        training_units = sorted(dev.iloc[train_indices]["unit"].astype(int).unique().tolist())
        validation_units = sorted(dev.iloc[validation_indices]["unit"].astype(int).unique().tolist())
        fold_assignments.append({
            "fold": fold_number, "train_units": json.dumps(training_units),
            "validation_units": json.dumps(validation_units),
            "n_train_windows": len(train_indices), "n_validation_windows": len(validation_indices),
        })

    cv_records = []
    for name in MODEL_NAMES:
        for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
            training = dev.iloc[train_indices]
            validation = dev.iloc[validation_indices]
            start = perf_counter()
            model = _fit(_new_model(name, seed), training, features)
            fit_seconds = perf_counter() - start
            predictions = validation[["unit", "cycle", "row_end", "target"]].copy()
            predictions["prediction"] = model.predict(validation[features])
            metrics = evaluate_predictions(predictions)
            for engine_metric in metrics["cycles"]["per_engine"]:
                cv_records.append({
                    "model": name, "fold": fold_number,
                    "train_units": fold_assignments[fold_number - 1]["train_units"],
                    "validation_units": fold_assignments[fold_number - 1]["validation_units"],
                    "fit_seconds": fit_seconds, **engine_metric,
                })

    cv_results = pd.DataFrame(cv_records)
    cv_summary = []
    for name in MODEL_NAMES:
        rows = cv_results[cv_results["model"] == name]
        cv_summary.append({
            "model": name,
            "macro_engine_cycle_mae": float(rows["mae"].mean()),
            "macro_engine_cycle_rmse": float(rows["rmse"].mean()),
            "n_heldout_engines": int(rows["unit"].nunique()),
            "n_folds": int(rows["fold"].nunique()),
        })
    # This decision is fixed before any test predictions are generated.
    # Stable model order provides a deterministic simple-model-first tie break.
    selected_model = min(cv_summary, key=lambda row: row["macro_engine_cycle_rmse"])["model"]

    artifacts: dict[str, Any] = {"models": {}}
    cv_path = output_dir / "cv_results.csv"
    cv_results.to_csv(cv_path, index=False)
    artifacts["cv_results_csv"] = str(cv_path)
    fold_path = output_dir / "cv_fold_assignments.csv"
    pd.DataFrame(fold_assignments).to_csv(fold_path, index=False)
    artifacts["cv_fold_assignments_csv"] = str(fold_path)

    test_metrics = {}
    all_window_predictions = []
    all_cycle_predictions = []
    for name in MODEL_NAMES:
        start = perf_counter()
        model = _fit(_new_model(name, seed), dev, features)
        fit_seconds = perf_counter() - start
        model_path = output_dir / f"model_{name}.joblib"
        joblib.dump(model, model_path)
        artifacts["models"][name] = str(model_path)
        metadata = [column for column in ["split", "unit", "cycle", "row_end", "flight_class", "target"] if column in test]
        predictions = test[metadata].copy()
        start = perf_counter()
        predictions["prediction"] = model.predict(test[features])
        predict_seconds = perf_counter() - start
        predictions["model"] = name
        predictions["selected_by_dev_cv"] = name == selected_model
        test_metrics[name] = {
            **evaluate_predictions(predictions),
            "fit_seconds": float(fit_seconds), "predict_seconds": float(predict_seconds),
        }
        all_window_predictions.append(predictions)
        cycles = aggregate_cycle_predictions(predictions)
        cycles["model"] = name
        cycles["selected_by_dev_cv"] = name == selected_model
        all_cycle_predictions.append(cycles)

    window_predictions = pd.concat(all_window_predictions, ignore_index=True)
    cycle_predictions = pd.concat(all_cycle_predictions, ignore_index=True)
    parquet_path = output_dir / "test_window_predictions.parquet"
    csv_path = output_dir / "test_window_predictions.csv"
    cycle_path = output_dir / "test_cycle_predictions.csv"
    window_predictions.to_parquet(parquet_path, index=False)
    window_predictions.to_csv(csv_path, index=False)
    cycle_predictions.to_csv(cycle_path, index=False)
    artifacts.update({
        "test_window_predictions_parquet": str(parquet_path),
        "test_window_predictions_csv": str(csv_path),
        "test_cycle_predictions_csv": str(cycle_path),
    })
    artifacts.update(_write_figures(cycle_predictions, test_metrics, selected_model, output_dir))
    metrics_path = output_dir / "metrics.json"
    artifacts["metrics_json"] = str(metrics_path)
    result = {
        "selected_model": selected_model,
        "selection_metric": "dev_cv_macro_engine_cycle_rmse",
        "feature_columns": features, "seed": int(seed),
        "development_units": sorted(DEV_UNITS), "test_units": sorted(TEST_UNITS),
        "evaluation_notes": {
            "cycle_aggregation": "Mean of all window predictions within an engine/flight cycle; available at flight end.",
            "macro_average": "Unweighted mean of per-engine MAE or RMSE; cycles equally weighted within each engine.",
            "fit_weights": "Relative weights 1 / windows_in_engine_cycle / cycles_in_engine, normalized to mean 1 (sum = number of fitting windows). Both stages use only the current fitting partition.",
            "selection": "Three-fold GroupKFold on development engines only; test metrics never select the model.",
            "prediction_policy": "Raw predictions are retained without clipping or target transformation.",
            "model_scope": "All three candidate models are refitted on all development engines and evaluated once on test engines.",
        },
        "cv_summary": cv_summary, "test_metrics": test_metrics, "artifacts": artifacts,
    }
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result
