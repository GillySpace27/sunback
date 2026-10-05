"""infra/pat_expiry.py with a fake Secrets Manager client (SB-7). No AWS."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra"))
import pat_expiry  # noqa: E402

NOW = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)


class Secrets:
    def __init__(self, tags=None):
        self.tags = tags or []
        self.calls = []

    def describe_secret(self, SecretId):
        self.calls.append(("describe_secret", SecretId))
        return {"Name": SecretId, "Tags": self.tags, "VersionIdsToStages": {"old-1": ["AWSCURRENT"]}}

    def put_secret_value(self, SecretId, SecretString):
        self.calls.append(("put_secret_value", SecretId, SecretString))
        return {"VersionId": "new-2"}

    def tag_resource(self, SecretId, Tags):
        self.calls.append(("tag_resource", SecretId, Tags))

    def __getattr__(self, name):
        raise AssertionError(f"unexpected Secrets Manager call: {name}")


def tagged(days):
    return Secrets([{"Key": "pat-expires", "Value": (NOW.date() + timedelta(days=days)).isoformat()}])


@pytest.mark.parametrize("days,code", [(5, 1), (14, 1), (15, 0), (-2, 1)])
def test_expiry_window(days, code):
    assert pat_expiry.main([], client=tagged(days), now=NOW) == code
    assert pat_expiry.days_until_expiry(client=tagged(days), now=NOW) == days


def test_missing_or_bad_tag_is_unchecked(capsys):
    assert pat_expiry.main(["--json"], client=Secrets(), now=NOW) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "UNCHECKED"
    bad = Secrets([{"Key": "pat-expires", "Value": "soon"}])
    assert pat_expiry.main([], client=bad, now=NOW) == 3
