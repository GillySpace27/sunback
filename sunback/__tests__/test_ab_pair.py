"""SB-20: the daily 171 A/B integration pair, offline on synthetic NRT frames."""
import json

import numpy as np
import pytest
from astropy.io import fits

from sunback.__tests__.fixtures.make_fits import make_synthetic_aia_fits
from sunback.__tests__.test_aws_putter_guards import fake_s3  # noqa: F401 (pytest fixture)
from sunback.run import run_ab_pair as ab


def _frames(tmp_path, n, shape=(64, 64), hit=True):
    """n synthetic 171 frames named like NRT files, 3 minutes apart; one carries a cosmic-ray hit."""
    paths = []
    for i in range(n):
        minute = 3 * i
        path = tmp_path / f"AIA20260928_12{minute:02d}00_0171.fits"
        make_synthetic_aia_fits(path, shape=shape, date_obs=f"2026-09-28T12:{minute:02d}:00.000", seed=i)
        paths.append(path)
    if hit:
        with fits.open(paths[-2], mode="update") as hdul:
            hdul[1].data[shape[0] // 2, shape[1] // 2] = 60000.0
    return [str(p) for p in paths]


def _plain(fits_path, out_png):
    """Stand-in renderer: copies nothing, writes a tiny file, so the halves can be tested without sunkit-image."""
    with open(out_png, "wb") as fh:
        fh.write(b"png")
    return out_png


def test_both_halves_share_the_newest_frame_and_record_their_settings(tmp_path):
    doc, files = ab.ab_pair(_frames(tmp_path, 15), str(tmp_path / "out"), render=_plain)
    assert (doc["current"]["frames"], doc["current"]["method"]) == (5, "median")
    assert (doc["candidate"]["frames"], doc["candidate"]["method"]) == (15, "sigclip")
    assert doc["current"]["t_rec"] == doc["candidate"]["t_rec"] == "2026-09-28T12:42:00Z"
    assert doc["source"]["newest"] == "AIA20260928_124200_0171.fits"
    assert doc["box"] == [28, 28, 36, 36]
    assert [k for _, k, _ in files] == ["lab/ab_171_current.png", "lab/ab_171_candidate.png", "lab/ab_171.json"]
    assert json.loads((tmp_path / "out" / "ab_171.json").read_text()) == doc
    assert "not a calibrated radiance" in doc["note"] and "computed" in doc["note"]


def test_the_candidate_is_quieter_and_rejects_the_hit(tmp_path):
    doc, _ = ab.ab_pair(_frames(tmp_path, 15), str(tmp_path / "out"), render=_plain)
    assert doc["candidate"]["std"] < doc["current"]["std"]
    with fits.open(tmp_path / "out" / "ab_171_candidate.fits") as hdul:
        assert hdul[1].data[32, 32] < 2000.0  # the 60000 hit is clipped, not averaged in


def test_fewer_frames_than_asked_are_recorded_not_widened(tmp_path):
    doc, _ = ab.ab_pair(_frames(tmp_path, 8, hit=False), str(tmp_path / "out"), render=_plain)
    assert (doc["candidate"]["frames"], doc["candidate"]["frames_asked"]) == (8, 15)
    assert doc["source"]["frames_available"] == 8


def test_box_std_is_the_numpy_std_inside_the_box():
    data = np.zeros((16, 16))
    data[6:10, 6:10] = np.arange(16).reshape(4, 4)
    assert ab.centre_box(data.shape) == [7, 7, 9, 9]
    assert ab.box_std(data, [6, 6, 10, 10]) == pytest.approx(np.std(np.arange(16)))


def test_the_real_rhef_render_writes_two_images(tmp_path):
    doc, files = ab.ab_pair(_frames(tmp_path, 6, hit=False), str(tmp_path / "out"))
    for path, key, _ in files[:2]:
        assert path.endswith(".png") and (tmp_path / "out" / key.split("/")[-1]).exists()


@pytest.mark.parametrize("prefix", ["", "staging/"])
def test_upload_goes_under_the_prefix(fake_s3, tmp_path, monkeypatch, prefix):  # noqa: F811
    monkeypatch.setenv("SUNBACK_PREFIX", prefix)
    _, files = ab.ab_pair(_frames(tmp_path, 5, hit=False), str(tmp_path / "out"), render=_plain)
    ab.upload(files)
    assert [k for _, k, _ in fake_s3.uploads] == [prefix + "lab/ab_171_current.png",
                                                  prefix + "lab/ab_171_candidate.png", prefix + "lab/ab_171.json"]


def test_cli_without_frames_writes_nothing(tmp_path, capsys):
    assert ab.main(["--frames-dir", str(tmp_path), "--out", str(tmp_path / "o"), "--no-upload"]) == 1
    assert not (tmp_path / "o").exists()

