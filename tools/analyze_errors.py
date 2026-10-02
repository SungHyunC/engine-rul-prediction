"""Descriptive error audit of the frozen final model; never fits a model."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "output"
    card = json.loads((output / "final/model_card.json").read_text(encoding="utf-8"))
    path = {"cnn_1d": output / "extended/cnn/test_cycle_predictions.csv",
            "sensitivity_winner": output / "extended/sensitivity/test_cycle_predictions.csv"}.get(card["selected_candidate"])
    if path:
        frame = pd.read_csv(path)
    else:
        frame = pd.read_csv(output / "results/test_cycle_predictions.csv")
        frame = frame.loc[frame.model == card["config"]["model"]].copy()
    frame["error"] = frame.prediction - frame.target
    frame["absolute_error"] = frame.error.abs()
    frame["life_stage"] = pd.cut(frame.target, [-np.inf, 10, 30, np.inf],
                                 labels=["RUL <= 10", "10 < RUL <= 30", "RUL > 30"])
    rows = []
    for (unit, stage), group in frame.groupby(["unit", "life_stage"], observed=True):
        rows.append({"unit": int(unit), "life_stage": str(stage), "n_cycles": len(group),
                     "mae": float(group.absolute_error.mean()), "rmse": float(np.sqrt((group.error ** 2).mean())),
                     "bias_pred_minus_true": float(group.error.mean())})
    per_engine_stage = pd.DataFrame(rows)
    per_engine_stage.to_csv(output / "final/error_by_engine_and_life_stage.csv", index=False)
    per_engine_stage.groupby("life_stage", as_index=False).agg(
        macro_engine_mae=("mae", "mean"), macro_engine_rmse=("rmse", "mean"),
        macro_engine_bias=("bias_pred_minus_true", "mean"), n_engines=("unit", "nunique"), n_cycles=("n_cycles", "sum")
    ).to_csv(output / "final/error_by_life_stage.csv", index=False)
    frame.sort_values("absolute_error", ascending=False).head(15).to_csv(output / "final/worst_cycles.csv", index=False)
    result = {"selected_candidate": card["selected_candidate"], "n_cycles": len(frame),
              "negative_predictions": int((frame.prediction < 0).sum()),
              "maximum_absolute_error": float(frame.absolute_error.max()),
              "stage_definition": "Post-hoc diagnostic bins from true RUL: <=10, (10,30], >30. Not model inputs or selection criteria.",
              "scope": "Descriptive audit of previously inspected test engines; no new validation claim"}
    (output / "final/error_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
