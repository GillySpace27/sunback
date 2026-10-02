"""Parked CI config (review of SB-13): dependabot.yml waits in ci-pending/ until
requirements-reducer.txt exists. Stdlib only: regexes over the YAML text."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARKED = ROOT / "ci-pending" / "dependabot.yml"


def test_dependabot_is_parked_not_active():
    assert not (ROOT / ".github" / "dependabot.yml").exists()
    assert not (ROOT / ".github" / "dependabot.yaml").exists()
    assert PARKED.exists()
    assert (ROOT / "ci-pending" / "README.md").read_text(encoding="utf-8").count("dependabot.yml") >= 1


def test_parked_dependabot_config_is_still_well_formed():
    text = PARKED.read_text(encoding="utf-8")
    assert re.search(r"^version: 2$", text, re.M)
    ecosystems = re.findall(r"^  - package-ecosystem: (\S+)$", text, re.M)
    assert ecosystems == ["pip", "github-actions", "docker"]
    assert len(re.findall(r"^      interval: weekly$", text, re.M)) == 3
