"""The SDO credit line is Gilly's decided sentence (2026-10-12, overview Q18), word for word."""
from pathlib import Path

from sunback.putter import AwsPutter

DECIDED = "Imagery courtesy of NASA/SDO and the AIA science team."
HANDOFF = Path(__file__).resolve().parents[2] / "docs" / "handoffs" / "SB-9-website.md"


def test_putter_credit_is_the_decided_sentence():
    assert AwsPutter.CREDIT == DECIDED


def test_website_handoff_carries_the_same_sentence():
    text = HANDOFF.read_text(encoding="utf-8")
    assert f"`{DECIDED}`" in text
    assert f'const CREDIT = "{DECIDED}";' in text
