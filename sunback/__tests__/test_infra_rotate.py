"""infra/rotate_dispatch_pat.py with fake AWS and gh (SB-7). Nothing real is called."""
import json
import sys
from pathlib import Path

from sunback.__tests__.test_infra_pat import NOW, Secrets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra"))
import rotate_dispatch_pat as rotate  # noqa: E402

TOKEN = "github_pat_" + "FAKE_FOR_TESTS_ONLY"  # not a real token shape; built at runtime


class Lambda:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        return {"StatusCode": 200, **({"FunctionError": self.error} if self.error else {})}


def answers(*items):
    it = iter(items)
    return lambda prompt="": next(it)


def test_declining_changes_nothing(capsys):
    sm, lam = Secrets(), Lambda()
    code = rotate.run([], sm=sm, lam=lam, ask=answers("2027-01-15", "no"), secret_prompt=lambda p: TOKEN,
                      gh=lambda: [], now=lambda: NOW, sleep=lambda s: None)
    assert code == 2 and sm.calls == [] and lam.calls == []
    assert TOKEN not in capsys.readouterr().out


def test_bad_date_or_token_is_refused():
    cases = ((answers("2026-09-30"), TOKEN), (answers("someday"), TOKEN), (answers("2027-01-15"), "hunter2"))
    for ask, token in cases:
        sm = Secrets()
        assert rotate.run([], sm=sm, lam=Lambda(), ask=ask, secret_prompt=lambda p, t=token: t,
                          gh=lambda: [], now=lambda: NOW, sleep=lambda s: None) == 2
        assert sm.calls == []


def test_rotation_stores_tags_dispatches_and_finds_the_run(capsys):
    sm, lam = Secrets(), Lambda()
    polls = iter([[{"createdAt": "2026-10-01T14:00:00Z", "url": "old"}],
                  [{"createdAt": "2026-10-01T15:00:30Z", "url": "https://github.com/x/runs/9"}]])
    code = rotate.run([], sm=sm, lam=lam, ask=answers("2027-01-15", "rotate"), secret_prompt=lambda p: TOKEN,
                      gh=lambda: next(polls), now=lambda: NOW, sleep=lambda s: None)
    assert code == 0
    stored = TOKEN if rotate.SECRET_JSON_FIELD is None else json.dumps({rotate.SECRET_JSON_FIELD: TOKEN})
    assert sm.calls[1] == ("put_secret_value", "github-actions-dispatch-token", stored)
    assert sm.calls[2] == ("tag_resource", "github-actions-dispatch-token",
                           [{"Key": "pat-expires", "Value": "2027-01-15"}])
    assert lam.calls == [{"FunctionName": "sun-reducer-dispatcher", "InvocationType": "RequestResponse",
                          "Payload": b"{}"}]
    out = capsys.readouterr().out
    assert TOKEN not in out
    assert "OK: run https://github.com/x/runs/9" in out


def test_failed_dispatch_prints_the_rollback(capsys):
    code = rotate.run([], sm=Secrets(), lam=Lambda(error="Unhandled"), ask=answers("2027-01-15", "rotate"),
                      secret_prompt=lambda p: TOKEN, gh=lambda: [], now=lambda: NOW, sleep=lambda s: None)
    assert code == 1
    out = capsys.readouterr().out
    assert "--move-to-version-id old-1 --remove-from-version-id new-2" in out
