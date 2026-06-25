"""Tests for selecting the N most-recent NRT synoptic frames per wavelength."""
import pytest

from sunback.fetcher.nrt_listing import parse_frame_time, select_recent_frames


def _names(*specs):
    """specs: (HHMMSS, wave) -> 'AIA20260624_<HHMMSS>_<wave>.fits'."""
    return [f"AIA20260624_{t}_{w}.fits" for t, w in specs]


def test_parse_frame_time_orders_correctly():
    early = parse_frame_time("AIA20260624_200000_0171.fits")
    late = parse_frame_time("AIA20260624_200300_0171.fits")
    assert late > early


def test_selects_n_most_recent_per_wave():
    files = _names(
        ("200000", "0171"), ("200300", "0171"), ("200600", "0171"),
        ("200000", "0193"), ("200300", "0193"),
    )
    result = select_recent_frames(files, waves=["0171", "0193"], n=2)
    assert result["0171"] == _names(("200300", "0171"), ("200600", "0171"))
    assert result["0193"] == _names(("200000", "0193"), ("200300", "0193"))


def test_returns_newest_last_for_natural_time_order():
    files = _names(("200600", "0171"), ("200000", "0171"), ("200300", "0171"))
    result = select_recent_frames(files, waves=["0171"], n=3)
    assert result["0171"] == _names(
        ("200000", "0171"), ("200300", "0171"), ("200600", "0171")
    )


def test_filters_out_unwanted_wavelengths():
    files = _names(("200000", "0171"), ("200000", "4500"), ("200000", "1600"))
    result = select_recent_frames(files, waves=["0171"], n=3)
    assert list(result.keys()) == ["0171"]
    assert result["0171"] == _names(("200000", "0171"))


def test_handles_fewer_than_n_available():
    files = _names(("200000", "0171"))
    result = select_recent_frames(files, waves=["0171"], n=5)
    assert result["0171"] == _names(("200000", "0171"))


def test_ignores_malformed_filenames():
    files = _names(("200000", "0171")) + ["index.html", "README", "AIA_bad.fits"]
    result = select_recent_frames(files, waves=["0171"], n=3)
    assert result["0171"] == _names(("200000", "0171"))


def test_wave_with_no_frames_is_absent():
    files = _names(("200000", "0171"))
    result = select_recent_frames(files, waves=["0171", "0094"], n=2)
    assert "0094" not in result
