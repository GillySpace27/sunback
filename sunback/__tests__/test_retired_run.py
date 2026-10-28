"""sunback-run is only a notice that points to Heliogram (2026-10-28)."""
import tomllib
from pathlib import Path

from sunback.client import retired

ROOT = Path(__file__).resolve().parents[2]


def test_entry_point_is_the_notice():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["sunback-run"] == "sunback.client.retired:main"


def test_notice_names_heliogram_and_fails(capsys):
    assert retired.main([]) == 1
    err = capsys.readouterr().err
    assert "https://gilly.space/heliogram/" in err
    assert "superseded by Heliogram" in err
    assert "\u2014" not in err


def test_readme_carries_the_notice():
    readme = (ROOT / "README.md").read_text()
    assert "superseded by\n> [Heliogram](https://gilly.space/heliogram/)" in readme
