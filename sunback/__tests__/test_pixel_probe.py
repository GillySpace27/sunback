"""devtools/scripts/pixel_probe.py: compare mode, and a deterministic render where sunkit-image is installed (SB-13)."""

import json

import numpy as np
import pytest

from devtools.scripts import pixel_probe as pp


def _probe_dir(tmp_path, name, arr, png=b"png"):
    d = tmp_path / name
    d.mkdir()
    np.save(d / "rhef.npy", arr)
    (d / "probe.png").write_bytes(png)
    return d


def test_compare_equal(tmp_path, capsys):
    a = np.linspace(0, 1, 16).reshape(4, 4)
    assert pp.compare(_probe_dir(tmp_path, "a", a), _probe_dir(tmp_path, "b", a.copy())) == 0
    assert json.loads(capsys.readouterr().out) == {"max_abs_diff": 0.0, "same_data": True, "same_png": True}


def test_compare_reports_the_difference(tmp_path, capsys):
    a = np.linspace(0, 1, 16).reshape(4, 4)
    b = a.copy()
    b[2, 3] += 0.25
    assert pp.compare(_probe_dir(tmp_path, "a", a), _probe_dir(tmp_path, "b", b, png=b"other")) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["same_data"] is False and out["same_png"] is False
    assert out["max_abs_diff"] == pytest.approx(0.25)


def test_render_is_deterministic(tmp_path, capsys):
    pytest.importorskip("sunkit_image")
    pytest.importorskip("sunback.__tests__.fixtures.make_fits")
    assert pp.render(tmp_path / "one") == 0
    first = json.loads(capsys.readouterr().out)
    assert pp.render(tmp_path / "two") == 0
    second = json.loads(capsys.readouterr().out)
    assert (first["data_sha256"], first["png_sha256"]) == (second["data_sha256"], second["png_sha256"])
    assert first["nan_count"] < 256 * 256
