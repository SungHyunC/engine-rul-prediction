"""Small provenance and launcher checks; no real dataset or model fitting."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

import ncmapss_rul.__main__ as cli
from ncmapss_rul.modeling import OBSERVABLE_VARIABLES


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    """Exercise real extraction and manifests against a tiny synthetic HDF5."""
    source = tmp_path / "synthetic.h5"
    with h5py.File(source, "w") as handle:
        handle["W_var"] = np.asarray(OBSERVABLE_VARIABLES[:4], dtype="S")
        handle["X_s_var"] = np.asarray(OBSERVABLE_VARIABLES[4:], dtype="S")
        handle["A_var"] = np.asarray(["unit", "cycle", "Fc", "hs"], dtype="S")
        for split, unit in (("dev", 2), ("test", 11)):
            values = np.arange(8 * 18, dtype=np.float64).reshape(8, 18) + unit
            handle[f"W_{split}"] = values[:, :4]
            handle[f"X_s_{split}"] = values[:, 4:]
            handle[f"Y_{split}"] = np.full((8, 1), 5.0)
            handle[f"A_{split}"] = np.tile([unit, 1, 1, 1], (8, 1))
    config = {
        "dataset": str(source.resolve()), "window": 4, "stride": 4,
        "chunk_rows": 5, "workers": 1, "seed": 410,
        "benchmark_workers": [1, 2], "benchmark_chunk_rows": [5, 10],
        "benchmark_repeats": 1,
    }
    directory = tmp_path / "data" / "processed"
    cli.prepare(config, directory)
    return config, directory, source


def test_matching_prepared_features_are_accepted(prepared):
    config, directory, _ = prepared
    dev, test = cli.load_verified_features(config, directory)
    assert len(dev) == len(test) == 2
    assert set(dev["split"]) == {"dev"}
    assert set(test["split"]) == {"test"}
    pd.testing.assert_frame_equal(dev, pd.read_parquet(directory / "dev.parquet"))
    pd.testing.assert_frame_equal(test, pd.read_parquet(directory / "test.parquet"))


@pytest.mark.parametrize("setting", ["window", "stride"])
def test_preprocessing_configuration_drift_is_rejected(prepared, setting):
    config, directory, _ = prepared
    modified = dict(config, **{setting: config[setting] + 1})
    with pytest.raises(ValueError, match=f"different {setting}"):
        cli.load_verified_features(modified, directory)


def test_execution_settings_can_change_without_invalidating_features(prepared):
    config, directory, _ = prepared
    # Worker/chunk choices cannot change extracted feature values; the seed is
    # for model fitting rather than feature extraction.
    modified = dict(config, workers=8, chunk_rows=1, seed=999)
    dev, test = cli.load_verified_features(modified, directory)
    assert len(dev) == len(test) == 2


def test_same_size_source_byte_change_is_rejected(prepared):
    config, directory, source = prepared
    original_size = source.stat().st_size
    with h5py.File(source, "r+") as handle:
        handle["W_dev"][0, 0] += 1
    assert source.stat().st_size == original_size
    with pytest.raises(ValueError, match="Input HDF5 changed"):
        cli.load_verified_features(config, directory)


def test_different_source_path_is_rejected_even_for_identical_bytes(prepared):
    config, directory, source = prepared
    copy_path = source.with_name("different_source.h5")
    shutil.copyfile(source, copy_path)
    modified = dict(config, dataset=str(copy_path.resolve()))
    with pytest.raises(ValueError, match="different dataset"):
        cli.load_verified_features(modified, directory)


@pytest.mark.parametrize("split", ["dev", "test"])
@pytest.mark.parametrize("tamper", ["value", "row_order", "column_order", "feature_names", "dtype"])
def test_saved_parquet_value_or_schema_tampering_is_rejected(prepared, split, tamper):
    config, directory, _ = prepared
    path = directory / f"{split}.parquet"
    frame = pd.read_parquet(path)
    if tamper == "value":
        frame.loc[0, "alt__mean"] += 1
    elif tamper == "row_order":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif tamper == "column_order":
        names = list(frame.columns)
        left, right = names.index("alt__mean"), names.index("Mach__mean")
        names[left], names[right] = names[right], names[left]
        frame = frame[names]
    elif tamper == "feature_names":
        # Both names remain allowed model features: validation of the set of
        # names alone cannot detect silently swapped sensor meanings.
        frame = frame.rename(columns={"alt__mean": "Mach__mean", "Mach__mean": "alt__mean"})
    else:
        frame["unit"] = frame["unit"].astype("int32")
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match=f"{split} features differ"):
        cli.load_verified_features(config, directory)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_configuration_rejects_nonpositive_or_noninteger_workers(tmp_path, value):
    config = json.loads((PROJECT_ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    config["workers"] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        cli.load_config(path)


def test_configuration_requires_single_worker_baseline(tmp_path):
    config = json.loads((PROJECT_ROOT / "configs" / "default.json").read_text(encoding="utf-8"))
    config["benchmark_workers"] = [2, 4]
    path = tmp_path / "no_baseline.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="include 1"):
        cli.load_config(path)


@pytest.mark.parametrize("repeats", [0, -1])
def test_cli_rejects_invalid_repeat_count_before_reading_dataset(monkeypatch, repeats, capsys):
    monkeypatch.setattr(sys, "argv", ["ncmapss_rul", "benchmark", "--repeats", str(repeats)])
    monkeypatch.setattr(cli, "load_config", lambda _: pytest.fail("Invalid options must fail before reading data"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "--repeats must be a positive integer" in capsys.readouterr().err


def test_cli_inspection_resolves_output_from_callers_working_directory(prepared, tmp_path, monkeypatch):
    config, _, _ = prepared
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    caller = tmp_path / "different caller directory"
    caller.mkdir()
    monkeypatch.chdir(caller)
    monkeypatch.setattr(sys, "argv", [
        "ncmapss_rul", "inspect", "--config", str(config_path), "--output", "local output",
    ])
    cli.main()
    output = caller / "local output" / "data_inspection.json"
    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["splits"]["dev"]["rows"] == 8
    assert report["splits"]["test"]["unit_ids"] == [11]


@pytest.mark.skipif(os.name != "nt", reason="PowerShell launcher is Windows-specific")
@pytest.mark.parametrize("child_exit_code", [0, 7])
def test_powershell_all_launcher_sets_project_context_and_propagates_failures(tmp_path, child_exit_code):
    """Run the real launcher with an inert stand-in module, not the real data."""
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("No PowerShell executable available")
    project = tmp_path / "project directory with spaces"
    project.mkdir()
    shutil.copyfile(PROJECT_ROOT / "run.ps1", project / "run.ps1")
    package = project / "src" / "ncmapss_rul"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CLI_PROBE_JSON']).write_text(json.dumps({"
        "'argv': sys.argv[1:], 'cwd': os.getcwd(), 'pythonpath': os.environ.get('PYTHONPATH')}))\n"
        "sys.exit(int(os.environ['CLI_PROBE_EXIT']))\n",
        encoding="utf-8",
    )
    probe = tmp_path / "launcher_probe.json"
    caller = tmp_path / "caller"
    caller.mkdir()
    environment = dict(os.environ, CLI_PROBE_JSON=str(probe), CLI_PROBE_EXIT=str(child_exit_code))
    process = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(project / "run.ps1"),
         "-Stage", "all", "-PythonPath", sys.executable, "-Repeats", "2"],
        cwd=caller, env=environment, capture_output=True, timeout=20,
    )
    assert probe.is_file(), process.stderr.decode(errors="replace")
    result = json.loads(probe.read_text(encoding="utf-8"))
    assert result["argv"] == ["all", "--repeats", "2"]
    assert Path(result["cwd"]).resolve() == project.resolve()
    assert Path(result["pythonpath"]).resolve() == (project / "src").resolve()
    assert (process.returncode == 0) == (child_exit_code == 0)
