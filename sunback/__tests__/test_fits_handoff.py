"""SB-17: the FITS behind every AIA picture, offline."""
import gzip

import numpy as np
import pytest
from astropy.io import fits

from aws_lambda.video_builder import manifest
from sunback.__tests__.fixtures.make_fits import make_synthetic_aia_fits
from sunback.__tests__.test_aws_putter_guards import fake_s3  # noqa: F401 (pytest fixture)
from sunback.putter import fits_handoff
from sunback.putter.serve_keys import s3_fits_key


def test_fits_keys_for_the_nine_aia_singles_only():
    assert s3_fits_key("171") == "fits/aia_171_latest.fits.gz"
    assert s3_fits_key("94") == "fits/aia_94_latest.fits.gz"
    assert s3_fits_key("1600") == "fits/aia_1600_latest.fits.gz"
    for pid in manifest.AIA_IDS:
        assert s3_fits_key(pid) == manifest.fits_key(pid)
        assert manifest.product_from_1k_key(s3_fits_key(pid)) is None   # never fires the Lambda
    for pid in ("rainbow", "composite_uv", "dem"):
        with pytest.raises(KeyError):
            s3_fits_key(pid)


def test_gz_holds_the_integrated_frame_and_header_but_no_rhef_frames(tmp_path):
    src = make_synthetic_aia_fits(tmp_path / "AIAsynoptic0171.fits", shape=(32, 32))
    with fits.open(src) as hdul:
        hdul.append(fits.ImageHDU(np.ones((32, 32), np.float32), name="RHEF(LEV1P5)"))  # as the reducer appends
        hdul.writeto(tmp_path / "with_rhef.fits")
    with fits.open(tmp_path / "with_rhef.fits") as hdul:   # the compressed frame as stored (lossy quantized)
        frame = np.array(hdul[1].data)
    out = fits_handoff.write_integrated_gz(str(tmp_path / "with_rhef.fits"), str(tmp_path / "a.fits.gz"))
    with open(out, "rb") as fp:
        assert fp.read(2) == b"\x1f\x8b"   # gzip magic
    with gzip.open(out) as gz, fits.open(gz) as hdul:
        assert len(hdul) == 2
        assert hdul[1].data.shape == (32, 32)
        np.testing.assert_array_equal(hdul[1].data, frame)
        assert hdul[1].header["WAVELNTH"] == 171 and hdul[1].header["CTYPE1"] == "HPLN-TAN"
        assert "T_REC" in hdul[1].header


@pytest.mark.parametrize("prefix", ["", "staging/"])
def test_publish_writes_one_fits_per_aia_single_found(fake_s3, tmp_path, monkeypatch, prefix):  # noqa: F811
    monkeypatch.setenv("SUNBACK_PREFIX", prefix)
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    for wave in ("0171", "1600"):
        make_synthetic_aia_fits(fits_dir / f"AIAsynoptic{wave}.fits", wave=wave)
    keys = fits_handoff.publish_integrated_fits(str(fits_dir), str(tmp_path / "work"))
    assert keys == [prefix + "fits/aia_171_latest.fits.gz", prefix + "fits/aia_1600_latest.fits.gz"]
    assert {x["ContentType"] for _, _, x in fake_s3.uploads} == {"application/gzip"}
    assert {x["ACL"] for _, _, x in fake_s3.uploads} == {"public-read"}


def test_publish_with_nothing_to_do_writes_nothing(fake_s3, tmp_path):  # noqa: F811
    assert fits_handoff.publish_integrated_fits(str(tmp_path), str(tmp_path / "w")) == []
    assert fits_handoff.publish_integrated_fits(None, str(tmp_path / "w")) == []
    assert fake_s3.uploads == []


def test_fragment_names_the_fits_and_its_size_only_when_given():
    integ = {"frames": 5, "method": "median"}
    base = manifest.build_manifest_fragment("171", updated="t", frame_count=1, integration=integ)
    assert "fits" not in base and "fits_bytes" not in base
    full = manifest.build_manifest_fragment("171", updated="t", frame_count=1, integration=integ,
                                            fits=manifest.fits_key("171"), fits_bytes=4321)
    assert manifest.validate_fragment(full) == []
    assert manifest.validate_fragment(dict(full, fits_bytes="4 kB")) == ["fits_bytes: expected int, got str"]


def test_lambda_names_the_fits_only_when_the_reducer_wrote_it(monkeypatch, tmp_path):
    from sunback.__tests__ import lambda_harness as lh

    fake = lh.FakeS3()
    handler = lh.load_handler(monkeypatch, fake, lh.fake_ffmpeg(tmp_path))
    assert handler._fits_download("171") == (None, None)
    fake._store("fits/aia_171_latest.fits.gz", b"x" * 1234)
    assert handler._fits_download("171") == ("fits/aia_171_latest.fits.gz", 1234)
    assert handler._fits_download("171", "staging/") == (None, None)
    fake._store("fits/aia_rainbow_latest.fits.gz", b"x")
    assert handler._fits_download("rainbow") == (None, None)


def test_reducer_publishes_the_fits_after_the_runner(monkeypatch):
    import importlib

    rsg = importlib.import_module("sunback.run.run_server_github")
    calls = []

    class Runner:
        def __init__(self, params):
            pass

        def start(self):
            calls.append("runner")

    monkeypatch.setattr(rsg, "SingleRunner", Runner)
    monkeypatch.setattr(rsg, "publish_grid_overlay", lambda *a, **k: [])
    monkeypatch.setattr(rsg, "publish_raw_views", lambda *a, **k: [])
    monkeypatch.setattr(rsg, "publish_integrated_fits",
                        lambda d, w, settings=None: calls.append(("fits", settings.prefix)) or [])
    monkeypatch.setenv("SUNBACK_PREFIX", "staging/")
    rsg.run_server_github()
    assert calls == ["runner", ("fits", "staging/")]
