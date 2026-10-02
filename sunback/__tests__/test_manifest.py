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


def test_versioned_keys_and_stamp():
    from aws_lambda.video_builder.manifest import (
        compact_stamp, versioned_still_key, versioned_video_key)
    assert compact_stamp("frames/171/20260924T180309_1k.png") == "20260924T180309"
    assert compact_stamp("2026-09-24T18:03:09Z") == "20260924T180309"
    # ids with underscores and digits must not confuse the stamp
    assert compact_stamp("frames/composite_uv/20260924T180309_1k.png") == "20260924T180309"
    assert versioned_video_key("171", "20260924T180309") == "v/171/20260924T180309.mp4"
    assert versioned_still_key("dem", "20260924T180309") == "v/dem/20260924T180309.png"


def test_compact_stamp_refuses_garbage():
    import pytest
    from aws_lambda.video_builder.manifest import compact_stamp
    with pytest.raises(ValueError):
        compact_stamp("video/rhef_171_1k.mp4")


def test_fragment_old_readers_see_nothing_new_unless_given():
    base = build_manifest_fragment("171", updated="t", frame_count=1, integration={})
    assert not {"video_v", "still_v", "through"} & set(base)
    full = build_manifest_fragment("171", updated="t", frame_count=1, integration={},
                                   video_v="v/171/a.mp4", still_v="v/171/b.png", through="x")
    assert full["video"] == "video/rhef_171_1k.mp4"      # fixed key unchanged
    assert (full["video_v"], full["still_v"], full["through"]) == ("v/171/a.mp4", "v/171/b.png", "x")


def test_index_orders_by_products_and_drops_strangers():
    from aws_lambda.video_builder.manifest import build_index
    frags = [{"id": "304"}, {"id": "999"}, {"id": "171"}, {"id": "rainbow"}]
    idx = build_index(frags, generated="g")
    assert idx["generated"] == "g"
    assert [f["id"] for f in idx["products"]] == ["rainbow", "171", "304"]
    assert build_index([], generated="g")["products"] == []

# --- SB-8: status.json, index extras, staging prefix ----------------------------
from datetime import datetime, timezone

from aws_lambda.video_builder.manifest import (
    INDEX_EXTRA_KEYS,
    STAGING_PREFIX,
    STATUS_KEY,
    build_index,
    build_status,
    split_staging_prefix,
)


def _frag(pid, updated):
    return build_manifest_fragment(pid, updated=updated, frame_count=1, integration={})


def test_build_status_reports_ages_and_worst():
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
    frags = [_frag("193", "2026-09-28T11:40:00Z"), _frag("171", "2026-09-28T09:00:00Z")]
    status = build_status(frags, "2026-09-28T12:00:00Z", now)
    assert STATUS_KEY == "status.json"
    assert status == {
        "generated": "2026-09-28T12:00:00Z",
        "products": [
            {"id": "171", "updated": "2026-09-28T09:00:00Z", "age_s": 10800},
            {"id": "193", "updated": "2026-09-28T11:40:00Z", "age_s": 1200},
        ],
        "worst_age_s": 10800,
    }


def test_build_status_skips_unknown_and_unparseable():
    now = datetime(2026, 9, 28, 12, 0, 0, tzinfo=timezone.utc)
    frags = [{"id": "nope", "updated": "2026-09-28T11:00:00Z"}, _frag("94", "t")]
    assert build_status(frags, "g", now) == {"generated": "g", "products": [], "worst_age_s": None}


def test_build_index_extras_merge_but_never_override():
    assert INDEX_EXTRA_KEYS == {}
    doc = build_index([_frag("171", "t")], "g", extras={"reel": {"video": "video/reel_48h.mp4"}})
    assert doc["reel"] == {"video": "video/reel_48h.mp4"}
    assert build_index([], "g") == {"generated": "g", "products": []}
    import pytest
    with pytest.raises(ValueError):
        build_index([], "g", extras={"generated": "x"})


def test_split_staging_prefix():
    assert STAGING_PREFIX == "staging/"
    assert split_staging_prefix("staging/1k/rhef_171_1k.png") == ("staging/", "1k/rhef_171_1k.png")
    assert split_staging_prefix("1k/rhef_171_1k.png") == ("", "1k/rhef_171_1k.png")


def test_fragment_obs_window_is_optional():
    """SB-9: obs_start/obs_end appear only when known (readers that predate them see no change)."""
    plain = build_manifest_fragment("171", updated="2026-09-28T12:00:00Z", frame_count=1, integration={})
    assert "obs_start" not in plain and "obs_end" not in plain
    frag = build_manifest_fragment("171", updated="2026-09-28T12:00:00Z", frame_count=1, integration={},
                                   obs_start="2026-09-28T11:52:00Z", obs_end="2026-09-28T12:00:00Z")
    assert (frag["obs_start"], frag["obs_end"]) == ("2026-09-28T11:52:00Z", "2026-09-28T12:00:00Z")


# ---- the public contract schema (SU-9) ----
import pathlib  # noqa: E402

import pytest  # noqa: E402

from aws_lambda.video_builder import manifest as _manifest  # noqa: E402
from aws_lambda.video_builder.schema.validate import validate  # noqa: E402

_VB = pathlib.Path(__file__).resolve().parents[2] / "aws_lambda" / "video_builder"
_INTEGRATION = {"frames": 5, "method": "median"}


def _schema(name):
    return json.loads((_VB / "schema" / name).read_text(encoding="utf-8"))


def _fragment(pid, **versions):
    return build_manifest_fragment(pid, updated="2026-09-24T18:03:09Z", frame_count=144,
                                   integration=dict(_INTEGRATION), **versions)


def test_fragment_matches_schema():
    schema = _schema("manifest-fragment.schema.json")
    for p in PRODUCTS:
        assert validate(_fragment(p["id"]), schema) == [], p["id"]
    versioned = _fragment("171",
                          video_v=_manifest.versioned_video_key("171", "20260924T180309"),
                          still_v=_manifest.versioned_still_key("171", "20260924T180309"),
                          through="2026-09-24T18:03:09Z")
    assert validate(versioned, schema) == []


def test_index_matches_schema():
    schema = _schema("index.schema.json")
    index = _manifest.build_index([_fragment(p["id"]) for p in PRODUCTS], generated="2026-09-24T18:03:09Z")
    assert validate(index, schema) == []
    assert validate(_manifest.build_index([], generated="2026-09-24T18:03:09Z"), schema) == []


def test_schema_rejects_a_renamed_required_key():
    frag = _fragment("171")
    frag["img_1k"] = frag.pop("img1k")
    assert "$: missing required key 'img1k'" in validate(frag, _schema("manifest-fragment.schema.json"))
    index = {"generated": "2026-09-24T18:03:09Z", "products": [frag]}
    assert "$.products[0]: missing required key 'img1k'" in validate(index, _schema("index.schema.json"))


def test_schema_rejects_a_bad_versioned_key():
    frag = _fragment("171", video_v="video/rhef_171_1k.mp4", still_v="v/171/20260924T180309.png")
    assert any(e.startswith("$.video_v:") for e in validate(frag, _schema("manifest-fragment.schema.json")))


def test_index_schema_embeds_the_fragment_schema():
    fragment = _schema("manifest-fragment.schema.json")
    embedded = _schema("index.schema.json")["properties"]["products"]["items"]
    for annotation in ("$schema", "title", "description"):
        fragment.pop(annotation, None)
    assert embedded == fragment


def test_schema_agrees_with_manifest_constants():
    required = getattr(_manifest, "FRAGMENT_REQUIRED", None)
    optional = getattr(_manifest, "FRAGMENT_OPTIONAL", None)
    if required is None or optional is None:
        pytest.skip("SB-6 has not landed: manifest.FRAGMENT_REQUIRED is absent")
    schema = _schema("manifest-fragment.schema.json")
    assert sorted(schema["required"]) == sorted(required)
    assert sorted(set(schema["properties"]) - set(required)) == sorted(optional)


def test_schema_accepts_provenance_fields_and_the_reserved_hdr_video():
    schema = _schema("manifest-fragment.schema.json")
    frag = _fragment("171", obs_start="2026-09-24T17:58:09Z", obs_end="2026-09-24T18:02:57Z")
    assert validate(frag, schema) == []
    # decision A1 (Gilly, 2026-10-02): the HDR video field is hdr_video, an optional string
    assert validate(dict(frag, hdr_video="hdr/rhef_171_1k_hlg.mp4"), schema) == []
    assert any(e.startswith("$.hdr_video:") for e in validate(dict(frag, hdr_video=3), schema))
    assert "hdr_video" in _manifest.FRAGMENT_OPTIONAL


def test_schema_accepts_the_punch_product_id():
    # decision A7 (Gilly, 2026-10-02): the PUNCH product id is `punch`, not `punch_cam`
    schema = _schema("manifest-fragment.schema.json")
    frag = _fragment("171")
    frag["id"] = "punch"
    assert validate(frag, schema) == []
