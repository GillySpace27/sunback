"""SB-16: the daily Filter Lab, offline on a synthetic 171 FITS."""
import json
import itertools

import numpy as np
import pytest
import yaml
from PIL import Image

from sunback.__tests__.fixtures.make_fits import make_synthetic_aia_fits
from sunback.__tests__.test_aws_putter_guards import fake_s3  # noqa: F401 (pytest fixture)
from sunback.run import run_filter_lab as lab

NAMES = ["rhef", "upsilon", "nrgf", "fnrgf", "msgn", "wow"]


@pytest.fixture(scope="module")
def doc_and_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lab")
    fits_path = make_synthetic_aia_fits(tmp / "AIAsynoptic0171.fits", shape=(128, 128))
    return lab.run_filter_lab(str(fits_path), str(tmp / "out"), upload=False), tmp / "out"


def test_every_filter_is_a_tile_or_listed_as_skipped(doc_and_dir):
    doc, out = doc_and_dir
    assert [f for f, _ in lab.FILTERS] == NAMES
    assert sorted([t["name"] for t in doc["tiles"]] + [s["name"] for s in doc["skipped"]]) == sorted(NAMES)
    assert all(s["error"] for s in doc["skipped"])
    assert {"generated", "source", "sunkit_image_version", "display", "note", "tiles", "skipped"} <= set(doc)
    assert "not a calibrated radiance" in doc["note"]
    assert json.loads((out / "filter_lab.json").read_text()) == doc
    for t in doc["tiles"]:
        assert t["key"] == f"lab/filter_{t['name']}_171.png" and (out / f"filter_{t['name']}_171.png").exists()


def test_the_core_filters_run_here(doc_and_dir):
    doc, _ = doc_and_dir
    assert {"rhef", "upsilon", "nrgf"} <= {t["name"] for t in doc["tiles"]}


def test_tiles_differ_pairwise(doc_and_dir):
    doc, out = doc_and_dir
    imgs = {t["name"]: np.asarray(Image.open(out / f"filter_{t['name']}_171.png"), dtype=float) for t in doc["tiles"]}
    for a, b in itertools.combinations(imgs, 2):
        assert np.abs(imgs[a] - imgs[b]).mean() > 0, (a, b)


def test_a_broken_filter_is_skipped_not_fatal(tmp_path):
    fits_path = make_synthetic_aia_fits(tmp_path / "AIAsynoptic0171.fits", shape=(32, 32))

    def broken(smap):
        raise RuntimeError("boom")

    doc = lab.run_filter_lab(str(fits_path), str(tmp_path / "o"), upload=False,
                             filters=[("broken", broken), ("msgn", lab._msgn)])
    assert doc["skipped"] == [{"name": "broken", "error": "RuntimeError: boom"}]
    assert [t["name"] for t in doc["tiles"]] == ["msgn"]


@pytest.mark.parametrize("prefix", ["", "staging/"])
def test_upload_goes_under_the_prefix(fake_s3, tmp_path, monkeypatch, prefix):  # noqa: F811
    monkeypatch.setenv("SUNBACK_PREFIX", prefix)
    fits_path = make_synthetic_aia_fits(tmp_path / "AIAsynoptic0171.fits", shape=(32, 32))
    lab.run_filter_lab(str(fits_path), str(tmp_path / "o"), filters=[("msgn", lab._msgn)])
    assert [k for _, k, _ in fake_s3.uploads] == [prefix + "lab/filter_msgn_171.png", prefix + "lab/filter_lab.json"]


def test_cli_without_a_fits_writes_nothing(tmp_path, capsys):
    assert lab.main(["--fits-dir", str(tmp_path), "--out", str(tmp_path / "o"), "--no-upload"]) == 1
    assert not (tmp_path / "o").exists()


def test_workflow_runs_the_lab_at_most_daily_and_never_fails_the_run():
    from pathlib import Path
    wf = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "GitCloudRunHourly.yml"
    steps = yaml.safe_load(wf.read_text())["jobs"]["run-task"]["steps"]
    names = [s.get("name") for s in steps]
    step = steps[names.index("Daily Filter Lab (SB-16)")]
    assert names.index("Daily Filter Lab (SB-16)") == names.index("Execute the script") + 1
    assert step["continue-on-error"] is True
    assert "check_freshness.py --key lab/filter_lab.json" in step["run"] and "--threshold 82800" in step["run"]
    assert "python -m sunback.run.run_filter_lab" in step["run"]
