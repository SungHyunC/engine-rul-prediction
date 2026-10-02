"""Recompute extension evidence and verify the label-free final prediction path."""
from datetime import datetime, timezone
import json
from itertools import islice
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import pandas as pd
from ncmapss_rul.__main__ import write_json
from ncmapss_rul.data import iter_segments, read_names, validate_layout
from ncmapss_rul.experiments import rank_candidates
from ncmapss_rul.features import extract_features
from ncmapss_rul.inference import predict_hdf5
from ncmapss_rul.modeling import evaluate_predictions
from ncmapss_rul.neural import cache_sequences, predict_cnn


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    output = ROOT / "output"
    sens_dir, cnn_dir = output / "extended/sensitivity", output / "extended/cnn"
    sensitivity, cnn = read(sens_dir / "metrics.json"), read(cnn_dir / "metrics.json")
    ranked = rank_candidates(pd.read_csv(sens_dir / "cv_results.csv"))
    selected = sensitivity["selected_config"]
    best = ranked.iloc[0]
    for key in ("model", "window", "stride", "nonnegative"):
        assert selected[key] == best[key], key
    np.testing.assert_allclose(best.cv_macro_engine_cycle_rmse, sensitivity["cv_macro_engine_cycle_rmse"])
    cv = pd.read_csv(cnn_dir / "cv_results.csv")
    assert set(cv.unit) == {2, 5, 10, 16, 18, 20} and len(cv) == 6
    np.testing.assert_allclose(cv.rmse.mean(), cnn["cv_macro_engine_cycle_rmse"])
    history = pd.read_csv(cnn_dir / "training_history.csv")
    assert set(history.phase) == {"fold_1", "fold_2", "fold_3", "full_dev"}
    assert history.groupby("phase").epoch.max().eq(cnn["config"]["epochs"]).all()
    for directory, metrics in ((sens_dir, sensitivity), (cnn_dir, cnn)):
        predictions = pd.read_parquet(directory / "test_window_predictions.parquet")
        computed = evaluate_predictions(predictions)
        for level in ("windows", "cycles"):
            for key in ("macro_engine_mae", "macro_engine_rmse", "micro_mae", "micro_rmse"):
                np.testing.assert_allclose(computed[level][key], metrics["test_metrics"][level][key], rtol=1e-6, atol=1e-6)
    # Saved CNN independently reproduces its predictions, not just its metrics.
    dataset = ROOT / "data/raw/N-CMAPSS_DS02-006.h5"
    base_test = pd.read_parquet(ROOT / "data/processed/test.parquet")
    sequences, _ = cache_sequences(dataset, base_test, ROOT / "data/sequences")
    actual = predict_cnn(cnn_dir / "model.pt", sequences)
    expected = pd.read_parquet(cnn_dir / "test_window_predictions.parquet").prediction.to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    del sequences
    card = read(output / "final/model_card.json")
    winner = min(card["candidates"], key=lambda row: (row["cv_rmse"], row["candidate"]))
    assert card["selected_candidate"] == winner["candidate"]
    saved_final = pd.read_parquet(output / "predictions/window_predictions.parquet")
    chosen_source = {"cnn_1d": cnn_dir, "sensitivity_winner": sens_dir}.get(card["selected_candidate"])
    if chosen_source:
        reference = pd.read_parquet(chosen_source / "test_window_predictions.parquet")
    else:
        reference = pd.read_parquet(output / "results/test_window_predictions.parquet")
        reference = reference.loc[reference.model == card["config"]["model"]]
    np.testing.assert_array_equal(saved_final[["unit", "cycle", "row_end"]].to_numpy(), reference[["unit", "cycle", "row_end"]].to_numpy())
    np.testing.assert_allclose(saved_final.prediction, reference.prediction, rtol=1e-5, atol=1e-5)
    # An independent smoke input contains observable arrays and no Y, health
    # state or latent variables. Two complete flights exercise boundaries.
    smoke = ROOT / "data/inference_smoke.h5"
    with h5py.File(dataset, "r") as source:
        segments = list(islice(iter_segments(source, validate_layout(source, "test", require_target=False)), 2))
        stop = segments[-1].end
        original_names = read_names(source["A_var"])
        kept_names = [name for name in original_names if name.casefold() in {"unit", "cycle", "fc"}]
        kept_indices = [original_names.index(name) for name in kept_names]
        with h5py.File(smoke, "w") as target:
            for name in ("W_var", "X_s_var"):
                target[name] = source[name][()]
            target["A_var"] = np.asarray(kept_names, dtype="S")
            target["A_test"] = source["A_test"][:stop][:, kept_indices]
            for name in ("W_test", "X_s_test"):
                target[name] = source[name][:stop]
    smoke_manifest = predict_hdf5(smoke, output / "final/model_card.json", output / "predictions/unlabelled_smoke", workers=1)
    assert smoke_manifest["input_target_required"] is False and smoke_manifest["n_windows"] > 0
    result = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "sensitivity_candidates": len(ranked), "sensitivity_cv_fits": sensitivity["n_cv_model_fits"],
              "cnn_cv_engines": len(cv), "cnn_fixed_epochs": cnn["config"]["epochs"],
              "cnn_checkpoint_reload": "matches saved predictions within float32 tolerance",
              "final_candidate": card["selected_candidate"],
              "final_prediction_path": "Matches selected experiment without reading Y",
              "unlabelled_smoke_windows": smoke_manifest["n_windows"],
              "unlabelled_smoke_cycles": smoke_manifest["n_cycles"],
              "test_status": "Previously inspected test set; exploratory extension comparison"}
    write_json(output / "extended_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
