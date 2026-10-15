"""SB-15: the grid and limb overlay, offline.

Every geometry number below is synthetic (chosen for the test, not an AIA
value). The alignment test puts the Sun off centre so a vertical flip, a
crop or a scale error between the overlay and the still moves the limb by
hundreds of pixels, not by a rounding error.
"""
import json
import math

import numpy as np
import pytest
from astropy.io import fits
from PIL import Image

from aws_lambda.video_builder import manifest
from sunback.__tests__.test_aws_putter_guards import fake_s3  # noqa: F401 (pytest fixture)
from sunback.processor import grid_overlay as go
from sunback.putter.serve_keys import overlay_grid_key

SIZE = 1024
CDELT = 2.5               # synthetic arcsec per pixel
DSUN_M = 1.496e11         # synthetic observer distance
RSUN_M = 6.957e8          # synthetic solar radius, written as RSUN_REF
SUN_X, SUN_Y = 600, 400   # 0-based FITS pixel of disk centre (off centre on purpose)


def _header(crpix1=SUN_X + 1, crpix2=SUN_Y + 1, n=SIZE):
    hdr = fits.Header()
    for key, value in {
        "NAXIS": 2, "NAXIS1": n, "NAXIS2": n,
        "CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN", "CUNIT1": "arcsec", "CUNIT2": "arcsec",
        "CDELT1": CDELT, "CDELT2": CDELT, "CRPIX1": crpix1, "CRPIX2": crpix2,
        "CRVAL1": 0.0, "CRVAL2": 0.0, "CROTA2": 0.0,
        "DATE-OBS": "2026-09-28T12:00:00.000", "DSUN_OBS": DSUN_M, "RSUN_REF": RSUN_M,
        "HGLN_OBS": 0.0, "HGLT_OBS": 0.0, "TELESCOP": "SDO/AIA", "INSTRUME": "AIA_3",
        "DETECTOR": "AIA", "WAVELNTH": 171, "WAVEUNIT": "angstrom",
    }.items():
        hdr[key] = value
    return hdr


def _write_fits(path, header):
    data = np.zeros((header["NAXIS2"], header["NAXIS1"]), dtype=np.float32)
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data=data, header=header)]).writeto(path)
    return path


def _alpha(path):
    img = Image.open(path)
    assert img.size == (SIZE, SIZE) and img.mode == "RGBA"
    return np.asarray(img)[:, :, 3]


def test_overlay_is_a_transparent_1k_rgba_png(tmp_path):
    out = tmp_path / "grid_1k.png"
    assert go.render_grid_overlay(go.header_map(_header()), str(out)) is True
    alpha = _alpha(out)
    assert alpha[0, 0] == alpha[0, -1] == alpha[-1, 0] == alpha[-1, -1] == 0
    assert (alpha == 0).mean() > 0.9, "the background must stay transparent"
    assert alpha.max() > 200, "grid and limb lines must be opaque"


def test_limb_lands_on_the_disk_edge_of_the_still(tmp_path):
    """Limb only: its centre and radius match the still's disk to within 1.5 px.

    The still is the FITS array flipped top to bottom (ImageProcessorCV
    plot_aia_changed, np.flipud), so FITS row y is PNG row SIZE - 1 - y.
    Removing draw_limb leaves no opaque pixel and fails this test.
    """
    out = tmp_path / "limb.png"
    assert go.render_grid_overlay(go.header_map(_header()), str(out), grid=False) is True
    rows, cols = np.nonzero(_alpha(out) > 128)
    assert rows.size > 100, "no limb drawn"
    radius_px = math.degrees(math.asin(RSUN_M / DSUN_M)) * 3600 / CDELT  # computed
    centre_row, centre_col = SIZE - 1 - SUN_Y, SUN_X
    assert abs(rows.mean() - centre_row) < 1.5 and abs(cols.mean() - centre_col) < 1.5
    dist = np.hypot(rows - centre_row, cols - centre_col)
    assert abs(np.median(dist) - radius_px) < 1.5


def test_wrong_size_writes_nothing(tmp_path):
    out = tmp_path / "grid.png"
    assert go.render_grid_overlay(go.header_map(_header(n=512)), str(out)) is False
    assert not out.exists()


def test_render_from_a_fits_path(tmp_path):
    src = _write_fits(tmp_path / "AIAsynoptic0171.fits", _header())
    out = tmp_path / "grid.png"
    assert go.render_grid_overlay(str(src), str(out)) is True
    assert _alpha(out).max() > 200


def test_geometry_mismatch_is_reported_per_wave():
    ref = _header()
    assert go.geometry_mismatches({"0193": _header()}, ref) == {}
    found = go.geometry_mismatches({"0193": _header(crpix1=SUN_X + 4), "0211": _header()}, ref)
    assert list(found) == ["0193"] and "CRPIX1" in found["0193"]


def test_overlay_keys_never_fire_the_video_lambda():
    assert overlay_grid_key() == go.OVERLAY_KEY == "overlay/grid_1k.png"
    for key in (go.OVERLAY_KEY, go.OVERLAY_MANIFEST_KEY):
        assert manifest.product_from_1k_key(key) is None
        assert not key.startswith("1k/")


def test_index_folds_the_overlay_manifest_in_as_overlay_grid():
    assert manifest.INDEX_EXTRA_KEYS["overlay_grid"] == go.OVERLAY_MANIFEST_KEY
    assert manifest.INDEX_OPTIONAL["overlay_grid"] is dict
    body = go.overlay_manifest("2026-09-28T12:00:00Z")
    assert body == {"key": "overlay/grid_1k.png", "updated": "2026-09-28T12:00:00Z", "source_id": "171"}
    doc = manifest.build_index([], "2026-09-28T12:00:00Z", extras={"overlay_grid": body})
    assert doc["overlay_grid"]["key"] == overlay_grid_key()


@pytest.mark.parametrize("prefix", ["", "staging/"])
def test_publish_writes_two_keys_under_the_prefix(fake_s3, tmp_path, monkeypatch, prefix):  # noqa: F811
    monkeypatch.setenv("SUNBACK_PREFIX", prefix)
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    _write_fits(fits_dir / "AIAsynoptic0171.fits", _header())
    _write_fits(fits_dir / "AIAsynoptic0193.fits", _header(crpix1=SUN_X + 9))
    keys = go.publish_grid_overlay(str(fits_dir), str(tmp_path / "work"))
    assert keys == [prefix + "overlay/grid_1k.png", prefix + "manifest/overlay.json"]
    assert [(k, x["ContentType"]) for _, k, x in fake_s3.uploads] == [
        (prefix + "overlay/grid_1k.png", "image/png"), (prefix + "manifest/overlay.json", "application/json")]
    body = json.loads((tmp_path / "work" / "overlay.json").read_text())
    assert set(body) == {"key", "updated", "source_id"} and body["key"] == "overlay/grid_1k.png"


def test_publish_without_the_171_fits_writes_nothing(fake_s3, tmp_path):  # noqa: F811
    assert go.publish_grid_overlay(str(tmp_path), str(tmp_path / "work")) == []
    assert go.publish_grid_overlay(None, str(tmp_path / "work")) == []
    assert fake_s3.uploads == []


def test_index_with_overlay_grid_matches_the_index_schema():
    from aws_lambda.video_builder.schema.validate import validate

    schema = json.loads((manifest_dir() / "schema" / "index.schema.json").read_text())
    doc = manifest.build_index([], "2026-09-28T12:00:00Z",
                               extras={"overlay_grid": go.overlay_manifest("2026-09-28T12:00:00Z")})
    assert validate(doc, schema) == []
    assert validate(dict(doc, overlay_grid={"key": "1k/rhef_171_1k.png"}), schema) != []


def manifest_dir():
    from pathlib import Path
    return Path(manifest.__file__).resolve().parent


def test_reducer_publishes_the_overlay_after_the_runner(monkeypatch):
    import importlib

    rsg = importlib.import_module("sunback.run.run_server_github")
    calls = []

    class Runner:
        def __init__(self, params):
            self.params = params

        def start(self):
            calls.append("runner")

    def publish(fits_dir, work_dir, settings=None):
        calls.append(("overlay", fits_dir, settings.prefix))
        return []

    monkeypatch.setattr(rsg, "SingleRunner", Runner)
    monkeypatch.setattr(rsg, "publish_grid_overlay", publish)
    monkeypatch.setenv("SUNBACK_PREFIX", "staging/")
    rsg.run_server_github()
    assert calls[0] == "runner" and calls[1][0] == "overlay" and calls[1][2] == "staging/"
