"""SB-8: infra/snapshot.py records the health alarms, alert topic and failure destination, read-only."""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _snapshot():
    spec = importlib.util.spec_from_file_location("infra_snapshot", ROOT / "infra" / "snapshot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeCW:
    def describe_alarms(self, AlarmNames):
        return {"MetricAlarms": [{"AlarmName": n, "Namespace": "AWS/Lambda", "MetricName": "Errors",
                                  "StateValue": "OK", "StateUpdatedTimestamp": "volatile"} for n in AlarmNames]}


class FakeSNS:
    def list_topics(self, **kw):
        return {"Topics": [{"TopicArn": "arn:aws:sns:us-east-2:123456789012:other"},
                           {"TopicArn": "arn:aws:sns:us-east-2:123456789012:sun-pipeline-alerts"}]}

    def list_subscriptions_by_topic(self, TopicArn):
        return {"Subscriptions": [{"Protocol": "email", "Endpoint": "someone@example.org",
                                   "SubscriptionArn": "PendingConfirmation"}]}


class FakeLambda:
    class exceptions:
        class ResourceNotFoundException(Exception):
            pass

    def get_function_event_invoke_config(self, FunctionName):
        return {"MaximumRetryAttempts": 2, "LastModified": "volatile",
                "DestinationConfig": {"OnFailure": {"Destination": "arn:aws:s3:::sun-video-builder-failures"}}}


def test_alarms_keep_definition_not_state():
    snap = _snapshot().snapshot_alarms(FakeCW())
    assert [a["AlarmName"] for a in snap] == sorted(_snapshot().HEALTH_ALARMS)
    assert all("StateValue" not in a and "StateUpdatedTimestamp" not in a for a in snap)


def test_topic_email_is_redacted():
    topic = _snapshot().snapshot_alert_topic(FakeSNS())
    assert topic["exists"] and topic["Subscriptions"] == [
        {"Protocol": "email", "Endpoint": "<EMAIL>", "Confirmed": False}]


def test_failure_destination_recorded():
    cfg = _snapshot().snapshot_event_invoke_config(FakeLambda())
    assert cfg["DestinationConfig"]["OnFailure"]["Destination"] == "arn:aws:s3:::sun-video-builder-failures"
    assert "LastModified" not in cfg


def test_snapshot_still_has_no_write_calls():
    text = (ROOT / "infra" / "snapshot.py").read_text(encoding="utf-8")
    assert not re.search(r"\b(put_|create_|delete_|update_)", text)
