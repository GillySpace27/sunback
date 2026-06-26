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


# --- build_grid_sequence: uniform 20-min grid with hold-previous gap fill ---
from aws_lambda.video_builder.frame_queue import build_grid_sequence


def test_uniform_frames_map_to_themselves():
    keys = _q("20260624T200000", "20260624T202000", "20260624T204000")
    seq = build_grid_sequence(keys, cadence_s=1200, max_slots=144)
    assert seq == keys  # one per slot, in order


def test_gap_slot_holds_previous_frame():
    # 00:00, 00:20, 01:00  (missing 00:40)
    keys = _q("20260624T000000", "20260624T002000", "20260624T010000")
    seq = build_grid_sequence(keys, cadence_s=1200, max_slots=144)
    # 4 slots: 00:00, 00:20, 00:40(held=00:20), 01:00
    assert seq == _q("20260624T000000", "20260624T002000",
                     "20260624T002000", "20260624T010000")


def test_max_slots_keeps_most_recent_window():
    keys = _q("20260624T200000", "20260624T202000", "20260624T204000",
              "20260624T210000", "20260624T212000", "20260624T214000")  # 20-min apart
    seq = build_grid_sequence(keys, cadence_s=1200, max_slots=3)
    assert len(seq) == 3
    assert seq[-1] == keys[-1]                     # window ends at the newest
    assert seq == keys[-3:]                        # the most-recent 3 slots


def test_empty_returns_empty():
    assert build_grid_sequence([], cadence_s=1200, max_slots=144) == []


def test_single_frame_one_slot():
    keys = _q("20260624T200000")
    assert build_grid_sequence(keys, cadence_s=1200, max_slots=144) == keys


# --- select_stale_frames: time-based pruning (robust to double-firing) ---
from aws_lambda.video_builder.frame_queue import select_stale_frames


def test_stale_frames_older_than_window_are_pruned():
    # newest = 02:00; window 1h -> anything before 01:00 is stale
    keys = _q("20260624T000000", "20260624T003000", "20260624T013000",
              "20260624T020000")
    stale = select_stale_frames(keys, window_s=3600)
    assert stale == _q("20260624T000000", "20260624T003000")


def test_nothing_stale_within_window():
    keys = _q("20260624T013000", "20260624T020000")
    assert select_stale_frames(keys, window_s=3600) == []


def test_stale_empty_is_empty():
    assert select_stale_frames([], window_s=3600) == []
