"""Tests for CSV, JSON and HDF5 export of tracking results."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from ndt.export import export_to_csv
from ndt.export import export_to_hdf5
from ndt.export import export_to_json
from ndt.export.csv import load_from_csv
from ndt.export.hdf5 import append_to_hdf5
from ndt.export.hdf5 import load_from_hdf5
from ndt.export.json import export_metadata
from ndt.export.json import load_from_json


def make_results():
    a = pd.DataFrame(
        {
            "step": [0, 10, 20],
            "stable_rank": [1.5, 2.25, 3.0],
            "loss": [0.9, 0.5, 0.25],
        }
    )
    b = pd.DataFrame(
        {
            "step": [0, 10, 20],
            "stable_rank": [4.0, 4.5, 5.0],
            "loss": [0.9, 0.5, 0.25],
        }
    )
    return {"Linear_0": a, "Linear_1": b}


def assert_results_equal(loaded, original):
    assert set(loaded) == set(original)
    for name, df in original.items():
        pd.testing.assert_frame_equal(
            loaded[name].reset_index(drop=True), df.reset_index(drop=True), check_dtype=False
        )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


class TestCSV:
    def test_combined_roundtrip(self, tmp_path, capsys):
        results = make_results()
        path = tmp_path / "results.csv"
        export_to_csv(results, str(path))

        assert "Exported all layers" in capsys.readouterr().out
        raw = pd.read_csv(path)
        assert list(raw.columns) == ["layer", "step", "stable_rank", "loss"]
        assert len(raw) == 6
        assert list(raw["layer"]) == ["Linear_0"] * 3 + ["Linear_1"] * 3

        assert_results_equal(load_from_csv(str(path)), results)

    def test_export_does_not_mutate_input(self, tmp_path):
        results = make_results()
        export_to_csv(results, str(tmp_path / "r.csv"))
        assert "layer" not in results["Linear_0"].columns

    def test_separate_files_roundtrip(self, tmp_path):
        results = make_results()
        out_dir = tmp_path / "nested" / "dir"
        export_to_csv(results, str(out_dir), separate_files=True)

        assert sorted(p.name for p in out_dir.iterdir()) == ["Linear_0.csv", "Linear_1.csv"]
        assert_results_equal(load_from_csv(str(out_dir), separate_files=True), results)

    def test_empty_results_writes_nothing(self, tmp_path, capsys):
        path = tmp_path / "empty.csv"
        export_to_csv({}, str(path))
        assert not path.exists()
        assert "No data to export" in capsys.readouterr().out

    def test_load_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_from_csv(str(tmp_path / "missing.csv"))

    def test_load_separate_requires_directory(self, tmp_path):
        path = tmp_path / "file.csv"
        path.write_text("a\n1\n")
        with pytest.raises(ValueError, match="Expected directory"):
            load_from_csv(str(path), separate_files=True)

    def test_load_combined_requires_layer_column(self, tmp_path):
        path = tmp_path / "nolayer.csv"
        pd.DataFrame({"step": [0, 1]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="'layer' column"):
            load_from_csv(str(path))


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestJSON:
    def test_roundtrip_pretty(self, tmp_path, capsys):
        results = make_results()
        path = tmp_path / "results.json"
        export_to_json(results, str(path))

        assert "Exported results" in capsys.readouterr().out
        text = path.read_text()
        assert "\n  " in text  # indented
        data = json.loads(text)
        assert data["Linear_0"][1] == {"step": 10, "stable_rank": 2.25, "loss": 0.5}
        assert_results_equal(load_from_json(str(path)), results)

    def test_compact(self, tmp_path):
        path = tmp_path / "compact.json"
        export_to_json(make_results(), str(path), pretty=False)
        text = path.read_text()
        assert "\n" not in text
        assert_results_equal(load_from_json(str(path)), make_results())

    def test_load_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_from_json(str(tmp_path / "nope.json"))

    def test_load_non_dict(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="must be a dictionary"):
            load_from_json(str(path))

    @pytest.mark.parametrize("pretty", [True, False])
    def test_export_metadata(self, tmp_path, pretty, capsys):
        meta = {"model_name": "MLP", "sampling_frequency": 10, "layers": ["a", "b"]}
        path = tmp_path / "meta.json"
        export_metadata(meta, str(path), pretty=pretty)
        assert json.loads(path.read_text()) == meta
        assert ("\n" in path.read_text()) is pretty
        assert "Exported metadata" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# HDF5
# ---------------------------------------------------------------------------


class TestHDF5:
    def test_roundtrip_with_metadata(self, tmp_path, capsys):
        results = make_results()
        path = tmp_path / "results.h5"
        export_to_hdf5(results, str(path), metadata={"model": "MLP", "epochs": 3})

        assert "Exported results" in capsys.readouterr().out
        loaded, meta = load_from_hdf5(str(path))
        assert_results_equal(loaded, results)
        assert meta["model"] == "MLP"
        assert int(meta["epochs"]) == 3

        with h5py.File(path, "r") as f:
            assert f["Linear_0"]["step"].compression == "gzip"
            assert f["Linear_0"]["step"].compression_opts == 4
            assert list(f["Linear_0"].attrs["columns"]) == ["step", "stable_rank", "loss"]

    def test_no_metadata(self, tmp_path):
        path = tmp_path / "plain.h5"
        export_to_hdf5(make_results(), str(path))
        _, meta = load_from_hdf5(str(path))
        assert meta == {}

    @pytest.mark.parametrize("compression", ["lzf", None])
    def test_other_compressions(self, tmp_path, compression):
        path = tmp_path / "c.h5"
        export_to_hdf5(make_results(), str(path), compression=compression)
        with h5py.File(path, "r") as f:
            assert f["Linear_1"]["stable_rank"].compression == compression
        assert_results_equal(load_from_hdf5(str(path))[0], make_results())

    @pytest.mark.xfail(
        reason=(
            "src bug: export_to_hdf5 converts string columns to numpy unicode ('<U') "
            "which h5py cannot store; pandas>=3 string columns also fail"
        ),
        strict=True,
    )
    def test_string_column(self, tmp_path):
        df = pd.DataFrame({"step": [0, 1], "phase": ["warmup", "train"]})
        path = tmp_path / "s.h5"
        export_to_hdf5({"L": df}, str(path))
        loaded, _ = load_from_hdf5(str(path))
        assert [str(v) for v in loaded["L"]["phase"]] == ["warmup", "train"]

    def test_load_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_from_hdf5(str(tmp_path / "missing.h5"))

    def test_append(self, tmp_path, capsys):
        path = tmp_path / "a.h5"
        first = make_results()
        export_to_hdf5(first, str(path))
        second = {
            name: df.assign(step=df["step"] + 30, stable_rank=df["stable_rank"] * 2)
            for name, df in first.items()
        }
        append_to_hdf5(second, str(path))
        assert "Appended results" in capsys.readouterr().out

        loaded, _ = load_from_hdf5(str(path))
        for name in first:
            expected = pd.concat([first[name], second[name]], ignore_index=True)
            assert_results_equal({name: loaded[name]}, {name: expected})
        np.testing.assert_array_equal(loaded["Linear_0"]["step"], [0, 10, 20, 30, 40, 50])

    def test_append_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="export_to_hdf5"):
            append_to_hdf5(make_results(), str(tmp_path / "missing.h5"))

    def test_append_unknown_layer(self, tmp_path):
        path = tmp_path / "a.h5"
        export_to_hdf5(make_results(), str(path))
        with pytest.raises(ValueError, match="Layer Conv_0 not found"):
            append_to_hdf5({"Conv_0": make_results()["Linear_0"]}, str(path))

    def test_append_unknown_column(self, tmp_path):
        path = tmp_path / "a.h5"
        export_to_hdf5(make_results(), str(path))
        bad = pd.DataFrame({"grad_norm": [1.0]})
        with pytest.raises(ValueError, match="Column grad_norm not found"):
            append_to_hdf5({"Linear_0": bad}, str(path))
