"""Tests for building JSOC synoptic-NRT hour-bucket directory URLs."""
from datetime import datetime

from sunback.fetcher.nrt_listing import nrt_hour_dirs

BASE = "https://jsoc1.stanford.edu/data/aia/synoptic/nrt/"


def test_current_hour_dir_first():
    dirs = nrt_hour_dirs(datetime(2026, 6, 24, 20, 12), base_url=BASE, lookback_hours=0)
    assert dirs == [BASE + "2026/06/24/H2000/"]


def test_lookback_includes_previous_hour_newest_first():
    dirs = nrt_hour_dirs(datetime(2026, 6, 24, 20, 12), base_url=BASE, lookback_hours=1)
    assert dirs == [
        BASE + "2026/06/24/H2000/",
        BASE + "2026/06/24/H1900/",
    ]


def test_hour_rollover_to_previous_day():
    dirs = nrt_hour_dirs(datetime(2026, 6, 24, 0, 5), base_url=BASE, lookback_hours=1)
    assert dirs == [
        BASE + "2026/06/24/H0000/",
        BASE + "2026/06/23/H2300/",
    ]


def test_base_url_without_trailing_slash_is_handled():
    dirs = nrt_hour_dirs(datetime(2026, 6, 24, 20, 12), base_url=BASE.rstrip("/"), lookback_hours=0)
    assert dirs == [BASE + "2026/06/24/H2000/"]
