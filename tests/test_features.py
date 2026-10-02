"""Small independent numeric oracles exercise chunking and temporal isolation."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from ncmapss_rul.data import inspect_hdf5
from ncmapss_rul.features import STATS, extract_features, feature_columns


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "synthetic.h5"
    groups = [(2, 1, 1, 9), (2, 2, 1, 8), (5, 1, 2, 10), (10, 1, 3, 5)]
    names = [f"w{i}" for i in range(4)] + [f"sensor{i}" for i in range(14)]
    with h5py.File(path, "w") as handle:
        handle["W_var"] = np.asarray(names[:4], dtype="S")[:, None]
        handle["X_s_var"] = np.asarray(names[4:], dtype="S")
        # Reordered metadata prevents positional assumptions about A.
        handle["A_var"] = np.asarray(["Fc", "hs", "cycle", "unit"], dtype="S")
        for split, shift in (("dev", 0), ("test", 10000)):
            values, metadata = [], []
            for group_index, (unit, cycle, fc, count) in enumerate(groups):
                t = np.arange(count, dtype=float)
                values.append(t[:, None] * (1 + np.arange(18)) + group_index * 100 + shift)
                metadata.extend([[fc, 1, cycle, unit] for _ in range(count)])
            array = np.concatenate(values)
            handle[f"W_{split}"] = array[:, :4]
            handle[f"X_s_{split}"] = array[:, 4:]
            handle[f"Y_{split}"] = (1000 + shift + np.arange(len(array), dtype=float))[:, None]
            handle[f"A_{split}"] = np.asarray(metadata, dtype=float)
            handle[f"T_{split}"] = np.full((len(array), 10), 999999.)
            handle[f"X_v_{split}"] = np.full((len(array), 14), 999999.)
    return path


def test_correct_statistics_causal_labels_and_boundaries(source):
    frame = extract_features(source, window=4, stride=3, chunk_rows=5)
    # Independent endpoint enumeration: windows reset at every flight boundary.
    assert frame.row_end.tolist() == [3, 6, 12, 15, 20, 23, 26, 30]
    assert frame.unit.tolist() == [2, 2, 2, 2, 5, 5, 5, 10]
    assert frame.cycle.tolist() == [1, 1, 2, 2, 1, 1, 1, 1]
    assert frame.flight_class.tolist() == [1, 1, 1, 1, 2, 2, 2, 3]
    np.testing.assert_array_equal(frame.target, 1000 + frame.row_end)
    with h5py.File(source, "r") as handle:
        for index, end in enumerate(frame.row_end):
            w = handle["W_dev"][end - 3:end + 1, 0]
            expected = [w.mean(), w.std(ddof=0), w.min(), w.max(), np.polyfit(np.arange(4), w, 1)[0]]
            np.testing.assert_allclose(frame.loc[index, [f"w0__{stat}" for stat in STATS]].to_numpy(float), expected, rtol=1e-6)
    assert frame[feature_columns(frame)].dtypes.eq(np.dtype("float32")).all()
    assert frame.target.dtype == np.float32
    assert frame.attrs["audit"]["candidate_windows"] == 8
    assert frame.attrs["audit"]["output_rows"] == 8


@pytest.mark.parametrize("chunk_rows", [1, 3, 4, 7, 120000])
@pytest.mark.parametrize("stride", [1, 3, 4, 7])
def test_chunk_size_invariance_and_stride_gaps(source, chunk_rows, stride):
    expected = extract_features(source, window=4, stride=stride, chunk_rows=120000)
    actual = extract_features(source, window=4, stride=stride, chunk_rows=chunk_rows)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for key in ("candidate_windows", "dropped_nonfinite_windows", "incomplete_tail_samples"):
        assert actual.attrs["audit"][key] == expected.attrs["audit"][key]


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_windows_spawn_worker_equivalence(source, workers):
    expected = extract_features(source, window=4, stride=3, chunk_rows=5)
    actual = extract_features(source, window=4, stride=3, chunk_rows=5, workers=workers)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_split_separation_and_no_forbidden_predictors(source):
    dev = extract_features(source, window=4, stride=4)
    test = extract_features(source, split="test", window=4, stride=4)
    assert set(test.split) == {"test"}
    np.testing.assert_array_equal(test.target - dev.target, np.full(len(test), 10000))
    names = [f"w{i}" for i in range(4)] + [f"sensor{i}" for i in range(14)]
    assert feature_columns(dev) == [f"{name}__{stat}" for name in names for stat in STATS]
    assert len(feature_columns(dev)) == 90
    assert not {"unit", "cycle", "flight_class", "target", "hs", "row_end"} & set(feature_columns(dev))


def test_future_values_cannot_change_earlier_features(source):
    before = extract_features(source, window=4, stride=1, chunk_rows=3)
    cutoff = 6
    with h5py.File(source, "r+") as handle:
        for name in ("W_dev", "X_s_dev", "Y_dev"):
            handle[name][cutoff:] = -999999
    after = extract_features(source, window=4, stride=1, chunk_rows=3)
    pd.testing.assert_frame_equal(before[before.row_end < cutoff], after[after.row_end < cutoff], check_exact=True)


def test_nonfinite_policy_and_audit(source):
    with h5py.File(source, "r+") as handle:
        handle["W_dev"][1, 0] = np.nan  # Drops the window ending at 3.
        handle["Y_dev"][6, 0] = np.inf  # Drops the window ending at 6.
        handle["Y_dev"][10, 0] = np.nan  # Non-endpoint label is not an input.
    frame = extract_features(source, window=4, stride=3, chunk_rows=2)
    assert frame.row_end.tolist() == [12, 15, 20, 23, 26, 30]
    assert frame.attrs["audit"]["candidate_windows"] == 8
    assert frame.attrs["audit"]["dropped_nonfinite_windows"] == 2


def test_reads_are_bounded_and_forbidden_arrays_never_read(source, monkeypatch):
    original = h5py.Dataset.__getitem__
    observed = []

    def guarded(dataset, selection):
        name = dataset.name.lstrip("/")
        if name.startswith(("X_v_", "T_")):
            pytest.fail(f"Forbidden array read: {name}")
        if name in {"W_dev", "X_s_dev", "Y_dev", "A_dev"}:
            assert isinstance(selection, slice)
            assert selection.stop - selection.start <= 3
            observed.append(name)
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    extract_features(source, window=4, stride=3, chunk_rows=3)
    assert set(observed) == {"W_dev", "X_s_dev", "Y_dev", "A_dev"}


def test_inspection_is_json_serializable_and_reads_only_metadata(source, monkeypatch):
    original = h5py.Dataset.__getitem__

    def guarded(dataset, selection):
        name = dataset.name.lstrip("/")
        if name.endswith(("_dev", "_test")) and not name.startswith("A_"):
            pytest.fail(f"Inspection read sensor or target values: {name}")
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    report = inspect_hdf5(source)
    json.dumps(report)
    assert report["splits"]["dev"]["unit_ids"] == [2, 5, 10]
    assert report["splits"]["dev"]["rows"] == 32
    assert report["splits"]["dev"]["units"][0]["cycles"] == [1, 2]
    assert report["splits"]["test"]["units"][1]["flight_classes"] == [2]
    assert report["datasets"]["X_s_dev"]["shape"] == [32, 14]


@pytest.mark.parametrize("option", ["window", "stride", "workers", "chunk_rows"])
def test_invalid_options(source, option):
    with pytest.raises(ValueError, match=option):
        extract_features(source, **{option: 0})


def test_incomplete_windows_return_typed_empty_frame(source):
    frame = extract_features(source, window=60, stride=60, chunk_rows=3)
    assert frame.empty
    assert len(feature_columns(frame)) == 90
    assert frame.target.dtype == np.float32
    assert frame.attrs["audit"]["incomplete_tail_samples"] == 32


def test_single_sample_window_and_flat_target(source):
    with h5py.File(source, "r+") as handle:
        y = handle["Y_dev"][:].reshape(-1)
        del handle["Y_dev"]
        handle["Y_dev"] = y
    frame = extract_features(source, window=1, stride=1, chunk_rows=2)
    assert len(frame) == 32
    assert frame.w0__std.eq(0).all()
    assert frame.w0__slope.eq(0).all()


def test_repeated_noncontiguous_flight_rejected(source):
    with h5py.File(source, "r+") as handle:
        handle["A_dev"][17:, 2] = 1
        handle["A_dev"][17:, 3] = 2
    with pytest.raises(ValueError, match="Noncontiguous repeated"):
        extract_features(source, window=4, chunk_rows=3)
