"""The public manifest schema: built fragments, captured fixtures and the validator agree (SB-6)."""
import json
from pathlib import Path

import pytest

from aws_lambda.video_builder import manifest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
INTEGRATION = {"frames": 5, "method": "median"}
VERSIONED = {"video_v": "v/171/20260928T120000.mp4", "still_v": "v/171/20260928T120000.png",
             "through": "2026-09-28T12:00:00Z"}


def _fragment(pid="171", **versions):
    return manifest.build_manifest_fragment(pid, updated="2026-09-28T12:00:00Z", frame_count=144,
                                            integration=dict(INTEGRATION), **versions)


def _fixture(name):
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{name} was not captured (SB-4)")
    text = path.read_text(encoding="utf-8")
    if "PLACEHOLDER" in text:  # SB-4 shipped stand-ins built offline; only a real capture is a contract witness
        pytest.skip(f"{name} is a PLACEHOLDER, not captured from the bucket (SB-4 Task 2 Step 3)")
    return json.loads(text)


def test_fragment_without_versions_has_exactly_the_required_keys():
    frag = _fragment()
    assert set(frag) == set(manifest.FRAGMENT_REQUIRED)
    assert manifest.validate_fragment(frag) == []


def test_fragment_with_versions_adds_only_documented_optional_keys():
    frag = _fragment(**VERSIONED)
    assert set(frag) - set(manifest.FRAGMENT_REQUIRED) == set(VERSIONED)
    assert set(VERSIONED) <= set(manifest.FRAGMENT_OPTIONAL)
    assert manifest.validate_fragment(frag) == []


@pytest.mark.parametrize("pid", [p["id"] for p in manifest.PRODUCTS])
def test_every_product_builds_a_valid_fragment(pid):
    assert manifest.validate_fragment(_fragment(pid, **VERSIONED)) == []


def test_validator_names_each_problem():
    frag = _fragment()
    del frag["video"]
    frag["frame_count"] = "144"
    frag["extra"] = 1
    frag["integration"] = {"frames": True}
    assert manifest.validate_fragment(frag) == [
        "missing required key 'video'",
        "frame_count: expected int, got str",
        "unknown key 'extra'",
        "integration.frames: expected int, got bool",
        "integration: missing 'method'",
    ]


def test_validator_rejects_unknown_ids_and_non_dicts():
    assert manifest.validate_fragment(dict(_fragment(), id="4500")) == ["id: unknown product id '4500'"]
    assert manifest.validate_fragment([]) == ["fragment: expected dict, got list"]


def test_captured_171_fragment_is_valid():
    assert manifest.validate_fragment(_fixture("manifest_171.json")) == []


def test_captured_index_is_valid():
    index = _fixture("manifest_index.json")
    allowed = {"generated", "products"} | set(manifest.INDEX_OPTIONAL)
    assert sorted(set(index) - allowed) == []
    assert isinstance(index["generated"], str)
    for frag in index["products"]:
        assert manifest.validate_fragment(frag) == [], frag.get("id")


CONTRACT = Path(__file__).resolve().parents[2] / "aws_lambda" / "video_builder" / "CONTRACT.md"


def test_contract_md_names_every_key_field_and_id():
    text = CONTRACT.read_text(encoding="utf-8")
    keys = [manifest.img1k_key("<id>"), manifest.thumb_key("<id>"), manifest.video_key("<id>"),
            manifest.manifest_key("<id>"), manifest.INDEX_KEY,
            manifest.versioned_video_key("<id>", "<stamp>"), manifest.versioned_still_key("<id>", "<stamp>"),
            "video/rhef_tscan.mp4", "image_times.txt", "image_times_readable.txt"]
    fields = sorted(set(manifest.FRAGMENT_REQUIRED) | set(manifest.FRAGMENT_OPTIONAL)
                    | set(manifest.INDEX_OPTIONAL) | {"generated", "products", "frames", "method"})
    ids = [p["id"] for p in manifest.PRODUCTS]
    assert [n for n in keys + fields + ids if f"`{n}`" not in text] == []
    assert manifest.IMMUTABLE in text
