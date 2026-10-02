"""The offline fixtures load, carry the keys the pipeline reads, and need no network."""
import json
import pathlib

import pytest
from astropy.io import fits

from sunback.__tests__.fixtures.make_fits import make_synthetic_aia_fits
from sunback.fetcher.nrt_integrate import read_image_data

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
FRAGMENT_KEYS = {"id", "label", "thumb", "img1k", "video", "updated", "frame_count", "integration"}


def test_make_fits_writes_synoptic_layout(tmp_path):
    p = make_synthetic_aia_fits(tmp_path / "AIAsynoptic0171.fits")
    assert p.exists()
    assert read_image_data(str(p)).shape == (64, 64)
    with fits.open(p) as hdul:
        assert hdul[0].data is None
        hdr = hdul[1].header
    for key in ("T_REC", "T_OBS", "DATE-OBS", "WAVELNTH", "X0_MP", "Y0_MP", "R_SUN"):
        assert key in hdr, key
    assert hdr["WAVELNTH"] == 171


def test_make_fits_is_deterministic(tmp_path):
    a = make_synthetic_aia_fits(tmp_path / "a.fits", seed=3)
    b = make_synthetic_aia_fits(tmp_path / "b.fits", seed=3)
    assert (read_image_data(str(a)) == read_image_data(str(b))).all()


def test_make_fits_opens_as_sunpy_map(tmp_path):
    sunpy_map = pytest.importorskip("sunpy.map")
    m = sunpy_map.Map(str(make_synthetic_aia_fits(tmp_path / "m.fits", wave="0193")))
    assert m.data.shape == (64, 64)
    assert int(m.wavelength.value) == 193


def test_manifest_171_fixture_has_fragment_keys():
    frag = json.loads((FIXTURES / "manifest_171.json").read_text())
    assert frag["id"] == "171"
    assert FRAGMENT_KEYS <= set(frag)


def test_manifest_index_fixture_lists_171():
    path = FIXTURES / "manifest_index.json"
    if not path.exists():
        pytest.skip("manifest/index.json was not public when the fixtures were captured")
    index = json.loads(path.read_text())
    assert {"generated", "products"} <= set(index)
    assert "171" in [p["id"] for p in index["products"]]
