"""Tests for per-product manifest fragments served to the landing page."""
import json

from aws_lambda.video_builder.manifest import build_manifest_fragment, PRODUCTS


def test_products_list_is_twelve_cards():
    ids = [p["id"] for p in PRODUCTS]
    assert ids == ["rainbow", "171", "193", "211", "304", "335", "94", "131",
                   "1600", "1700", "composite_uv", "dem"]


def test_fragment_has_all_asset_keys():
    frag = build_manifest_fragment(
        "171", updated="2026-06-24T20:20:00Z", frame_count=144,
        integration={"frames": 3, "method": "median"},
    )
    assert frag["id"] == "171"
    assert frag["label"] == "AIA 171 Å"
    assert frag["img1k"] == "1k/rhef_171_1k.png"
    assert frag["thumb"] == "thumb/rhef_171_thumb.png"
    assert frag["video"] == "video/rhef_171_1k.mp4"
    assert frag["updated"] == "2026-06-24T20:20:00Z"
    assert frag["frame_count"] == 144
    assert frag["integration"] == {"frames": 3, "method": "median"}


def test_rainbow_label_and_keys():
    frag = build_manifest_fragment("rainbow", updated="t", frame_count=1, integration={})
    assert "Rainbow" in frag["label"]
    assert frag["img1k"] == "1k/rhef_rainbow_1k.png"


def test_fragment_is_json_serializable():
    frag = build_manifest_fragment("94", updated="t", frame_count=10, integration={})
    assert json.loads(json.dumps(frag))["id"] == "94"


def test_unknown_product_raises():
    import pytest
    with pytest.raises(KeyError):
        build_manifest_fragment("999", updated="t", frame_count=0, integration={})


def test_product_from_1k_key():
    from aws_lambda.video_builder.manifest import product_from_1k_key
    assert product_from_1k_key("1k/rhef_171_1k.png") == "171"
    assert product_from_1k_key("1k/rhef_rainbow_1k.png") == "rainbow"
    # ids may contain underscores (e.g. composite_uv) — must not be dropped
    assert product_from_1k_key("1k/rhef_composite_uv_1k.png") == "composite_uv"
    assert product_from_1k_key("1k/rhef_1600_1k.png") == "1600"


def test_product_from_1k_key_rejects_non_1k():
    from aws_lambda.video_builder.manifest import product_from_1k_key
    assert product_from_1k_key("thumb/rhef_171_thumb.png") is None
    assert product_from_1k_key("video/rhef_171_1k.mp4") is None
