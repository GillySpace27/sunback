"""infra/diff.py against fake AWS clients (SB-7). No AWS, no network."""
import json
import re
import sys

from sunback.__tests__.test_infra_snapshot import INFRA, Fake, fake_clients

sys.path.insert(0, str(INFRA))
import diff  # noqa: E402
import snapshot  # noqa: E402


def test_diff_makes_no_write_calls():
    source = (INFRA / "diff.py").read_text()
    assert re.findall(r"put_|create_|delete_|update_", source) == []


def test_diff_is_0_when_live_matches_and_1_on_drift(tmp_path, capsys):
    declared = tmp_path / "declared"
    clients = fake_clients()
    clients["budgets"] = Fake(describe_budget={"Budget": {"BudgetName": "whole-account-monthly"}},
                              describe_notifications_for_budget={"Notifications": []})
    assert snapshot.main(["--out", str(declared)], clients=clients) == 0
    assert diff.main(["--declared", str(declared)], clients=clients) == 0
    clients["events"] = fake_clients(rule_state="DISABLED")["events"]
    capsys.readouterr()
    assert diff.main(["--declared", str(declared)], clients=clients) == 1
    out = capsys.readouterr().out
    assert '-    "State": "ENABLED"' in out and '+    "State": "DISABLED"' in out
    assert out.rstrip().endswith("DRIFT: 1 file(s) drifted, 0 unchecked")


def test_diff_without_declared_snapshot_is_unchecked(tmp_path, capsys):
    assert diff.main(["--declared", str(tmp_path / "declared"), "--json"], clients=fake_clients()) == 3
    assert json.loads(capsys.readouterr().out)["status"] == "UNCHECKED"
