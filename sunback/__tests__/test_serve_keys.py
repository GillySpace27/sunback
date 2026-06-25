"""Map real reducer PNG filenames -> served product ids + S3 keys."""
import pytest

from sunback.putter.serve_keys import (
    serve_id_for_local_png,
    s3_img_key,
    s3_thumb_key,
)


@pytest.mark.parametrize("name,expected", [
    ("DrGilly_0171_ups(rhef).png", "171"),
    ("DrGilly_0193_ups(rhef).png", "193"),
    ("DrGilly_0211_ups(rhef).png", "211"),
    ("DrGilly_0304_ups(rhef).png", "304"),
    ("DrGilly_0335_ups(rhef).png", "335"),
    ("DrGilly_0094_ups(rhef).png", "94"),
    ("DrGilly_0131_ups(rhef).png", "131"),
])
def test_euv_singles_map_to_card_ids(name, expected):
    assert serve_id_for_local_png(name) == expected


def test_headline_composite_maps_to_rainbow():
    assert serve_id_for_local_png("BGR_0171_0193_0211_ups(rhef).png") == "rainbow"


@pytest.mark.parametrize("name,expected", [
    ("DrGilly_1600_ups(rhef).png", "1600"),
    ("DrGilly_1700_ups(rhef).png", "1700"),
    ("BGR_1700_1600_0304_ups(rhef).png", "composite_uv"),
    ("C_isothermal.png", "dem"),
])
def test_extra_products_map(name, expected):
    assert serve_id_for_local_png(name) == expected


def test_unserved_things_return_none():
    for name in [
        "a_temp_video_small.mp4",              # video handled separately, not a still
        "image_times_readable.txt",
        "DrGilly_4500_ups(rhef).png",          # visible-light channel, not served
    ]:
        assert serve_id_for_local_png(name) is None, name


def test_full_path_is_accepted():
    assert serve_id_for_local_png("/x/y/renders/DrGilly_0304_ups(rhef).png") == "304"


def test_s3_key_builders():
    assert s3_img_key("171") == "1k/rhef_171_1k.png"
    assert s3_thumb_key("rainbow") == "thumb/rhef_rainbow_thumb.png"
