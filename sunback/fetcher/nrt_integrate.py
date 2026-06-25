"""Integrate several NRT frames of one wavelength into a synoptic-compatible FITS.

The downstream pipeline (RHEF/Upsilon/RainbowRGB) consumes one
``AIAsynoptic<wave>.fits`` per wavelength, structured like a single JSOC synoptic
file (empty primary + a 2D science HDU). This module collapses the N most-recent
NRT frames into exactly that structure, swapping the science data for the
time-integrated array while preserving the newest frame's header/metadata.
"""
from astropy.io import fits

from sunback.utils.time_integration import integrate_frames


def _science_hdu_index(hdul):
    """Index of the first HDU holding a 2D image (handles compressed FITS)."""
    for i, hdu in enumerate(hdul):
        if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
            return i
    raise ValueError("No 2D image HDU found")


def read_image_data(path):
    """Return the 2D science array from a (possibly compressed) FITS file."""
    with fits.open(path) as hdul:
        return hdul[_science_hdu_index(hdul)].data.copy()


def write_integrated_synoptic(frame_paths, out_path, method="median"):
    """Integrate ``frame_paths`` (oldest->newest) into ``out_path``.

    The newest frame (last in the list) supplies the header. Output mirrors the
    JSOC synoptic layout: empty primary + a CompImageHDU science array.
    """
    if not frame_paths:
        raise ValueError("write_integrated_synoptic requires at least one frame")

    frames = [read_image_data(p) for p in frame_paths]
    integrated = integrate_frames(frames, method=method)

    with fits.open(frame_paths[-1]) as hdul:  # newest frame for metadata
        science_header = hdul[_science_hdu_index(hdul)].header.copy()

    science_header["TINT_N"] = (len(frame_paths), "frames time-integrated")
    science_header["TINT_M"] = (method, "time-integration method")

    out = fits.HDUList([
        fits.PrimaryHDU(),
        fits.CompImageHDU(data=integrated, header=science_header),
    ])
    out.writeto(out_path, overwrite=True)
    return out_path
