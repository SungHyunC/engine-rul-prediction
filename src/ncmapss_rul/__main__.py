"""Command line entry point for repeatable DS02 experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Each parallel worker receives one numerical-library thread. This keeps worker
# count experiments from silently multiplying BLAS/OpenMP parallelism.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

from .data import inspect_hdf5
from .features import extract_features
from .modeling import train_evaluate

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def fingerprint(frame: pd.DataFrame) -> str:
    """Hash ordered column meanings, dtypes and ordered row values together."""
    schema = [(str(name), str(dtype)) for name, dtype in frame.dtypes.items()]
    digest = hashlib.sha256(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    dataset = Path(config["dataset"])
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    config["dataset"] = str(dataset.resolve())
    for name in ("window", "stride", "chunk_rows", "workers", "benchmark_repeats"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if 1 not in config["benchmark_workers"]:
        raise ValueError("benchmark_workers must include 1 for the speedup baseline")
    for name in ("benchmark_workers", "benchmark_chunk_rows"):
        if not config[name] or any(type(value) is not int or value < 1 for value in config[name]):
            raise ValueError(f"{name} must contain positive integers")
    return config


def prepare(config: dict, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    audits = {}
    for split in ("dev", "test"):
        print(f"Extracting {split} features with {config['workers']} workers", flush=True)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            frame = extract_features(config["dataset"], split=split,
                                     window=config["window"], stride=config["stride"],
                                     workers=config["workers"], chunk_rows=config["chunk_rows"])
        elapsed = time.perf_counter() - started
        audit = dict(frame.attrs.get("audit", {}))
        audit.update(elapsed_seconds=elapsed, sha256_rows=fingerprint(frame))
        frame.to_parquet(output / f"{split}.parquet", index=False)
        audits[split] = audit
        print(f"{split}: {len(frame):,} windows, {elapsed:.2f} s", flush=True)
    audits["config"] = config
    audits["fingerprint_version"] = "ordered-schema-dtypes-pandas-rows-v2"
    audits["dataset_sha256"] = file_sha256(config["dataset"])
    audits["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "feature_manifest.json", audits)
    return audits


def load_verified_features(config: dict, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((directory / "feature_manifest.json").read_text(encoding="utf-8"))
    for name in ("dataset", "window", "stride"):
        if manifest["config"][name] != config[name]:
            raise ValueError(f"Prepared features use a different {name}; run prepare again.")
    if manifest["dataset_sha256"] != file_sha256(config["dataset"]):
        raise ValueError("Input HDF5 changed since prepare; run prepare again.")
    frames = []
    for split in ("dev", "test"):
        frame = pd.read_parquet(directory / f"{split}.parquet")
        if fingerprint(frame) != manifest[split]["sha256_rows"]:
            raise ValueError(f"{split} features differ from their manifest; run prepare again.")
        frames.append(frame)
    return frames[0], frames[1]


def benchmark(config: dict, output: Path, repeats: int | None = None) -> dict:
    """Fresh processes; warmed OS cache; timing includes HDF5 scan and worker spawn."""
    output.mkdir(parents=True, exist_ok=True)
    repeats = repeats or config["benchmark_repeats"]
    settings = [(workers, config["chunk_rows"]) for workers in config["benchmark_workers"]]
    settings += [(config["workers"], chunk) for chunk in config["benchmark_chunk_rows"]]
    settings = list(dict.fromkeys(settings))
    rows = []
    reference_hash = None
    # Warm the OS file cache with one untimed-to-the-table full development pass.
    runs = [("warmup", 1, config["chunk_rows"], -1)]
    rng = np.random.default_rng(config["seed"])
    for repeat in range(repeats):
        for index in rng.permutation(len(settings)):
            workers, chunk = settings[int(index)]
            runs.append(("measured", workers, chunk, repeat))
    for number, (kind, workers, chunk, repeat) in enumerate(runs):
        result_path = output / "runs" / f"{number:03d}.json"
        result_path.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-m", "ncmapss_rul", "_benchmark_one",
                   "--dataset", config["dataset"], "--window", str(config["window"]),
                   "--stride", str(config["stride"]), "--workers", str(workers),
                   "--chunk-rows", str(chunk), "--result", str(result_path)]
        print(f"Benchmark {kind} repeat={repeat + 1}, workers={workers}, chunk={chunk}", flush=True)
        started = time.perf_counter()
        peak = 0
        log_path = result_path.with_suffix(".log")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            root_process = psutil.Process(process.pid)
            while process.poll() is None:
                rss = 0
                try:
                    members = [root_process] + root_process.children(recursive=True)
                except psutil.Error:
                    members = [root_process]
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except psutil.Error:
                        pass
                peak = max(peak, rss)
                time.sleep(.1)
        if process.returncode:
            raise RuntimeError(f"Benchmark failed; see {log_path}")
        row = json.loads(result_path.read_text(encoding="utf-8"))
        row.update(kind=kind, repeat=repeat, peak_process_tree_rss_bytes=peak,
                   process_wall_seconds=time.perf_counter() - started)
        if reference_hash is None:
            reference_hash = row["sha256_rows"]
        if row["sha256_rows"] != reference_hash:
            raise AssertionError("Feature outputs changed across benchmark configurations")
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / "benchmark_runs.csv", index=False)
    measured = pd.DataFrame([row for row in rows if row["kind"] == "measured"])
    summary = measured.groupby(["workers", "chunk_rows"], as_index=False).agg(
        median_seconds=("elapsed_seconds", "median"), min_seconds=("elapsed_seconds", "min"),
        max_seconds=("elapsed_seconds", "max"),
        peak_rss_mib=("peak_process_tree_rss_bytes", lambda values: float(values.max() / 2**20)),
        input_rows=("input_rows", "first"), output_windows=("output_windows", "first"))
    baseline = float(summary.loc[(summary.workers == 1) &
                                (summary.chunk_rows == config["chunk_rows"]), "median_seconds"].iloc[0])
    summary["speedup_vs_one_worker"] = baseline / summary.median_seconds
    summary["efficiency"] = summary.speedup_vs_one_worker / summary.workers
    summary["input_rows_per_second"] = summary.input_rows / summary.median_seconds
    summary.to_csv(output / "benchmark_summary.csv", index=False)
    info = {
        "dataset_split": "dev", "repeats": repeats, "config": config,
        "timing_scope": "HDF5 metadata scan, sensor reads, process spawning (including worker imports), feature calculations and result assembly. Excludes initial imports in the parent process, result hashing and saving.",
        "cache_policy": "One complete warmup; OS cache is not flushed. No cold-disk performance claim.",
        "memory_scope": "Peak sampled sum of worker process-tree RSS every 100 ms. Shared pages may be counted more than once; not physical memory consumption.",
        "correctness": "Every run has exactly the same ordered schema, dtype and pandas row fingerprint.",
        "fingerprint_version": "ordered-schema-dtypes-pandas-rows-v2",
        "platform": platform.platform(), "python": sys.version, "cpu_count_logical": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False), "total_memory_bytes": psutil.virtual_memory().total,
        "summary": summary.to_dict(orient="records"), "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "benchmark_manifest.json", info)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scaling = summary.loc[summary.chunk_rows == config["chunk_rows"]].sort_values("workers")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(scaling.workers, scaling.median_seconds, "o-", color="#175a72")
    axes[0].set(xlabel="Worker processes", ylabel="Median preprocessing time (s)", title="HDF5 preprocessing")
    axes[1].plot(scaling.workers, scaling.speedup_vs_one_worker, "o-", color="#175a72", label="Measured")
    axes[1].plot(scaling.workers, scaling.workers, "--", color="#888888", label="Ideal")
    axes[1].set(xlabel="Worker processes", ylabel="Speedup", title="Scaling on one PC")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(scaling.workers)
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "benchmark_scaling.png", dpi=180)
    plt.close(fig)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "prepare", "train", "benchmark", "all", "sensitivity", "cnn", "extended", "finalize", "predict", "full", "_benchmark_one"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-rows", type=int, default=120000)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--model-card", type=Path, default=ROOT / "output/final/model_card.json")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.result is not None:
        args.result = args.result.resolve()
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be a positive integer")
    if args.command == "_benchmark_one":
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            frame = extract_features(args.dataset, "dev", args.window, args.stride,
                                     args.workers, args.chunk_rows)
        elapsed = time.perf_counter() - started
        write_json(args.result, {
            "workers": args.workers, "chunk_rows": args.chunk_rows,
            "elapsed_seconds": elapsed, "output_windows": len(frame),
            "input_rows": frame.attrs["audit"]["input_rows"], "sha256_rows": fingerprint(frame),
        })
        return
    config = load_config(args.config)
    if args.dataset:
        config["dataset"] = str(Path(args.dataset).resolve())
    if not Path(config["dataset"]).is_file():
        parser.error("Dataset is missing. Run tools/download_data.py first.")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command in {"inspect", "all", "full"}:
        info = inspect_hdf5(config["dataset"])
        write_json(args.output / "data_inspection.json", info)
        for split, details in info["splits"].items():
            print(f"{split}: {details['rows']:,} rows, engines {details['unit_ids']}", flush=True)
    if args.command in {"prepare", "all", "full"}:
        prepare(config, ROOT / "data/processed")
    if args.command in {"train", "all", "full"}:
        dev, test = load_verified_features(config, ROOT / "data/processed")
        with threadpool_limits(limits=1):
            result = train_evaluate(dev, test, args.output / "results", seed=config["seed"])
        print(f"Selected by development CV: {result['selected_model']}", flush=True)
    if args.command in {"benchmark", "all", "full"}:
        benchmark(config, args.output / "benchmark", args.repeats)
    if args.command in {"sensitivity", "extended", "full"}:
        from .experiments import run_sensitivity
        with threadpool_limits(limits=1):
            run_sensitivity(config["dataset"], args.output / "extended/sensitivity", ROOT / "data/processed/sensitivity",
                            seed=config["seed"], workers=config["workers"], chunk_rows=config["chunk_rows"])
    if args.command in {"cnn", "extended", "full"}:
        from .neural import run_cnn
        dev, test = load_verified_features(config, ROOT / "data/processed")
        run_cnn(config["dataset"], dev, test, args.output / "extended/cnn", ROOT / "data/sequences",
                seed=config["seed"], epochs=12, batch_size=512)
    if args.command in {"finalize", "full"}:
        from .inference import finalize_model
        finalize_model(args.output)
    if args.command in {"predict", "full"}:
        from .inference import predict_hdf5
        card = args.output / "final/model_card.json" if args.command == "full" else args.model_card
        predict_hdf5(config["dataset"], card, args.output / "predictions", split=args.split,
                     workers=config["workers"], chunk_rows=config["chunk_rows"])


if __name__ == "__main__":
    main()
