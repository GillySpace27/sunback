"""Tests for the Lambda video-builder's 48h sliding-window frame queue."""
import pytest

from aws_lambda.video_builder.frame_queue import (
    frame_key_for,
    sorted_newest_last,
    select_frames_to_delete,
)


def _q(*stamps):
    return [f"frames/0171/{s}_1k.png" for s in stamps]


def test_frame_key_for_builds_expected_path():
    key = frame_key_for("0171", "2026-06-24T20:03:00Z")
    assert key == "frames/0171/20260624T200300_1k.png"


def test_sorted_newest_last_orders_by_timestamp():
    keys = _q("20260624T200600", "20260624T200000", "20260624T200300")
    assert sorted_newest_last(keys) == _q(
        "20260624T200000", "20260624T200300", "20260624T200600"
    )


def test_nothing_deleted_when_under_limit():
    keys = _q("20260624T200000", "20260624T200300")
    assert select_frames_to_delete(keys, max_frames=144) == []


def test_oldest_deleted_when_over_limit():
    keys = _q("20260624T200600", "20260624T200000", "20260624T200300")
    # keep newest 2 -> delete the oldest one
    assert select_frames_to_delete(keys, max_frames=2) == _q("20260624T200000")


def test_deletes_all_but_newest_n():
    keys = _q(*[f"20260624T2{i:02d}000" for i in range(10)])
    to_delete = select_frames_to_delete(keys, max_frames=3)
    assert len(to_delete) == 7
    # the 3 newest must NOT be in the delete list
    assert _q("20260624T207000", "20260624T208000", "20260624T209000") == [
        k for k in keys if k not in to_delete
    ]


def test_ignores_unparseable_keys_in_ordering():
    keys = _q("20260624T200300") + ["frames/0171/thumb.txt"]
    # junk is not counted as a frame to keep, and never returned for deletion here
    assert select_frames_to_delete(keys, max_frames=144) == []
