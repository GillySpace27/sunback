"""Round-trip tests: integrate N NRT frames into one synoptic-compatible FITS."""
import numpy as np
import pytest
from astropy.io import fits

from sunback.fetcher.nrt_integrate import write_integrated_synoptic, read_image_data


def _write_comp_fits(path, data, wavelnth=171, t_rec="2026-06-24T20:00:00Z"):
    """Mimic a JSOC synoptic file: empty primary + CompImageHDU science array."""
    hdr = fits.Header()
    hdr["WAVELNTH"] = wavelnth
    hdr["T_REC"] = t_rec
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.CompImageHDU(data=data.astype(np.float32), header=hdr),
    ])
    hdul.writeto(path, overwrite=True)


def test_read_image_data_handles_compressed(tmp_path):
    p = tmp_path / "in.fits"
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    _write_comp_fits(str(p), data)
    assert np.allclose(read_image_data(str(p)), data)


def test_median_integration_removes_spike_and_writes_synoptic(tmp_path):
    base = np.full((4, 4), 10.0, dtype=np.float32)
    spike = base.copy(); spike[1, 1] = 9999.0
    paths = []
    for i, arr in enumerate([base, spike, base]):
        p = tmp_path / f"AIA20260624_20{i:02d}00_0171.fits"
        _write_comp_fits(str(p), arr)
        paths.append(str(p))

    out = tmp_path / "AIAsynoptic0171.fits"
    write_integrated_synoptic(paths, str(out), method="median")

    result = read_image_data(str(out))
    assert result[1, 1] == pytest.approx(10.0)   # cosmic ray rejected
    assert result.shape == (4, 4)


def test_output_preserves_newest_header(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"AIA_{i}.fits"
        _write_comp_fits(str(p), np.ones((2, 2), dtype=np.float32),
                         t_rec=f"2026-06-24T20:0{i}:00Z")
        paths.append(str(p))
    out = tmp_path / "AIAsynoptic0171.fits"
    write_integrated_synoptic(paths, str(out), method="mean")

    with fits.open(str(out)) as hdul:
        # the science HDU carries the original metadata for downstream extraction
        hdr = next(h.header for h in hdul if h.data is not None and h.data.ndim == 2)
        assert hdr["WAVELNTH"] == 171
        assert hdr["T_REC"] == "2026-06-24T20:01:00Z"  # newest (last) frame
