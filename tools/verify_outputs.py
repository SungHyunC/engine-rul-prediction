"""Verify saved experimental outputs and record reproducibility evidence.

Run after prepare/train/benchmark/tests; this does not refit or select a model.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ncmapss_rul.__main__ import fingerprint, load_config, load_verified_features, write_json
from ncmapss_rul.modeling import evaluate_predictions, get_feature_columns


def main() -> None:
    config = load_config(ROOT / "configs/default.json")
    dev, test = load_verified_features(config, ROOT / "data/processed")
    acquired = json.loads((ROOT / "data/raw/source_manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((ROOT / "data/processed/feature_manifest.json").read_text(encoding="utf-8"))
    if acquired["sha256"] != prepared["dataset_sha256"]:
        raise AssertionError("Experiment source does not match the verified acquisition manifest")
    results = ROOT / "output/results"
    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    selected_by_cv = min(metrics["cv_summary"], key=lambda row: row["macro_engine_cycle_rmse"])["model"]
    if selected_by_cv != metrics["selected_model"]:
        raise AssertionError("Saved selection does not match the development-only metric")
    saved = pd.read_parquet(results / "test_window_predictions.parquet")
    checks = {}
    for name in metrics["test_metrics"]:
        predictions = saved.loc[saved.model == name].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            predictions[["unit", "cycle", "row_end", "target"]],
            test[["unit", "cycle", "row_end", "target"]].reset_index(drop=True),
        )
        # Only load models produced in this local workspace by this pipeline.
        model = joblib.load(results / f"model_{name}.joblib")
        with threadpool_limits(limits=1):
            current = model.predict(test[get_feature_columns(test)])
        np.testing.assert_allclose(current, predictions.prediction.to_numpy(), rtol=1e-6, atol=1e-6)
        recomputed = evaluate_predictions(predictions)
        for level in ("windows", "cycles"):
            for measure in ("macro_engine_mae", "macro_engine_rmse", "micro_mae", "micro_rmse"):
                # Ridge originally returns float32; concatenating with float64
                # candidates widens the saved predictions, changing rounding.
                np.testing.assert_allclose(recomputed[level][measure], metrics["test_metrics"][name][level][measure], rtol=1e-6, atol=1e-6)
        checks[name] = "Saved model predictions, identifiers and all aggregate metrics match"
    runs = pd.read_csv(ROOT / "output/benchmark/benchmark_runs.csv")
    if set(runs.sha256_rows) != {fingerprint(dev)}:
        raise AssertionError("Benchmark configurations differ from the prepared development features")
    measured = runs.loc[runs.kind == "measured"]
    counts = measured.groupby(["workers", "chunk_rows"]).size()
    expected_settings = {(workers, config["chunk_rows"]) for workers in config["benchmark_workers"]}
    expected_settings |= {(config["workers"], chunk) for chunk in config["benchmark_chunk_rows"]}
    if not (counts == config["benchmark_repeats"]).all() or set(counts.index) != expected_settings:
        raise AssertionError("Benchmark settings or repeat counts differ from the configuration")
    provenance = ROOT / "output/provenance"
    provenance.mkdir(exist_ok=True)
    for source in (ROOT / "data/raw/source_manifest.json", ROOT / "data/processed/feature_manifest.json"):
        shutil.copyfile(source, provenance / source.name)
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    paths += [ROOT / "configs/default.json", ROOT / "configs/extended.json", ROOT / "requirements-tested.txt", ROOT / "run.ps1", ROOT / "pyproject.toml"]
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "h5py", "scikit-learn", "pyarrow", "psutil", "matplotlib",
        "requests", "threadpoolctl", "pytest", "joblib", "scipy", "torch",
    )}
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_date_local": "2026-10-02", "student": "김성현", "student_id": "2021271250",
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "config": config, "original_baseline_selected_by_development_cv": selected_by_cv,
        "output_checks": checks,
        "benchmark_check": f"All {len(runs)} passes match current development feature schema, dtypes and ordered values",
        "test_results": "output/test_results.txt",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "note": "Verification only: no model was refitted and no test-driven selection was made.",
        "numerical_tolerance": "rtol=1e-6, atol=1e-6; accommodates float32 Ridge predictions widened in the multi-model output table",
    }
    final_card = ROOT / "output/final/model_card.json"
    if final_card.is_file():
        card = json.loads(final_card.read_text(encoding="utf-8"))
        result["final_selected_by_development_cv"] = card["selected_candidate"]
        result["final_model_sha256"] = card["model_sha256"]
        result["extension_verification"] = "output/extended_verification.json"
        result["test_reuse_disclosure"] = card["evaluation_disclosure"]
    write_json(ROOT / "output/execution_summary.json", result)
    print(f"Verified all three saved models, prediction metrics, and {len(runs)} benchmark passes.")


if __name__ == "__main__":
    main()
