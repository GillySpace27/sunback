"""The production FITS-to-PNG step (ImageProcessorCV) on a synthetic map (SB-11).

Runs the same class and settings run_server_github.py uses for the served
stills (png_frame_name ["ups(rhef)"]) on a 256 x 256 synthetic file and checks
the file name, location, shape and dtype. No pixel golden: the colours depend
on matplotlib and OpenCV builds, so a golden image would fail on version drift
rather than on a real change.
"""
import numpy as np
import pytest
from astropy.io import fits

from sunback.__tests__.fixtures.make_fits import make_synthetic_aia_fits

cv2 = pytest.importorskip("cv2")
pytest.importorskip("OpenImageIO")
pytest.importorskip("pytz")

# The shape of a real T_OBS (fractional seconds, no zone), as image_times.txt shows it.
T_OBS = "2026-09-28T12:00:00.000"


def make_rhef_upsilon_fits(path):
    """A synoptic-layout file plus an HDU named like the frame run_server_github.py renders (ups(rhef))."""
    make_synthetic_aia_fits(path, wave="0171", shape=(256, 256))
    with fits.open(path, mode="update") as hdul:
        data = hdul[1].data.astype(np.float32)
        hdul.append(fits.ImageHDU((data - data.min()) / (data.max() - data.min()),
                                  header=hdul[1].header.copy(), name="ups(rhef)"))
        for hdu in hdul[1:]:
            hdu.header["T_OBS"] = T_OBS
    return path


def test_imageprocessorcv_writes_the_served_png(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # Parameters roots sunback_data/ at the working directory
    from sunback.processor.ImageProcessorCV import ImageProcessorCV
    from sunback.science.parameters import Parameters

    fits_path = make_rhef_upsilon_fits(tmp_path / "AIAsynoptic0171.fits")
    p = Parameters()
    p.is_debug(True)
    p.do_one("rainbow", True)
    p.batch_name("sb11_render")
    p.do_parallel = False
    p.reprocess_mode(True)
    p.do_upsilon = True
    p.visualization_style = "threshold"
    p.png_frame_name = ["ups(rhef)"]
    p.rgb_frame = "rhef(lev1p5)"
    p.current_wave(171)

    ImageProcessorCV(params=p, rp=True).do_fits_function(str(fits_path))

    pngs = sorted(tmp_path.rglob("*.png"))
    assert [png.name for png in pngs] == ["DrGilly_0171_ups(rhef).png"]
    assert pngs[0].parent == tmp_path / "sunback_data" / "renders" / "sb11_render" / "rainbow" / "imgs" / "mod"
    img = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8
    assert img.std() > 0
