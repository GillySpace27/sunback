"""Runner failure behaviour (Gilly's decision on Q14, 2026-10-02: retry with a pause).

Run mode (``is_debug()`` False: the wallpaper client and sunback-serve) waits
``retry_pause_s`` (30 s) after a failed batch and tries again, up to
``retry_max`` (10) retries. When the retries are used up, the last error is
re-raised, so the process exits with it. Debug mode (the production reducer,
``run_server_github(debug=True)``) is unchanged: it raises on the first failure
and never sleeps. The sleep is injected (``Runner.sleep``), so no test waits.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from sunback.run.run import Runner


class FlakyRunner(Runner):
    def __init__(self, params, failures):
        super().__init__(params)
        self.failures = failures
        self.calls = 0
        self.pauses = []
        self.sleep = self.pauses.append  # injected clock: record, never wait

    def process(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"failure {self.calls}")


def params(debug, stop_after_one=True):
    return SimpleNamespace(is_debug=lambda: debug, stop_after_one=lambda: stop_after_one)


def test_run_mode_retries_ten_times_then_reraises_the_last_error():
    runner = FlakyRunner(params(debug=False), failures=50)
    with pytest.raises(RuntimeError, match="failure 11"):
        runner.start(verb=False)
    assert runner.calls == 11  # the first attempt plus 10 retries


def test_run_mode_pauses_30_seconds_between_attempts():
    runner = FlakyRunner(params(debug=False), failures=50)
    with pytest.raises(RuntimeError):
        runner.start(verb=False)
    assert runner.pauses == [30] * 10  # one pause per retry, none after the last failure


def test_run_mode_stops_retrying_after_success_on_the_third_attempt():
    runner = FlakyRunner(params(debug=False), failures=2)
    runner.start(verb=False)
    assert runner.calls == 3
    assert runner.pauses == [30, 30]


def test_run_mode_success_resets_the_retry_budget():
    # fails on calls 1..10, succeeds on call 11, then fails again: the budget is per run of failures
    class Pattern(FlakyRunner):
        def process(self):
            self.calls += 1
            if self.calls != 11 and self.calls <= 25:
                raise RuntimeError(f"failure {self.calls}")

    runner = Pattern(params(debug=False, stop_after_one=False), failures=0)
    outcomes = iter([False] * 100)
    runner.params.stop_after_one = lambda: next(outcomes)
    with pytest.raises(RuntimeError, match="failure 22"):
        runner.start(verb=False)
    assert runner.calls == 22  # 10 failures, success, then 11 failures (10 retries) and the raise


def test_run_mode_does_not_pause_when_nothing_fails():
    runner = FlakyRunner(params(debug=False), failures=0)
    runner.start(verb=False)
    assert (runner.calls, runner.pauses) == (1, [])


def test_debug_mode_raises_on_first_failure_and_never_sleeps():
    runner = FlakyRunner(params(debug=True), failures=1)
    with pytest.raises(RuntimeError, match="failure 1"):
        runner.start(verb=False)
    assert runner.calls == 1
    assert runner.pauses == []


def test_the_github_reducer_runs_in_debug_mode_by_default():
    """run_server_github() passes debug=True to Parameters.is_debug, which keeps the reducer on the raise-at-once path."""
    import importlib
    from unittest.mock import MagicMock

    gh = importlib.import_module("sunback.run.run_server_github")
    src = Path(gh.__file__).read_text()
    assert re.search(r"def run_server_github\([^)]*debug=True", src)
    assert "p.is_debug(debug)" in src
    p = MagicMock()
    started = []

    class RecordingRunner:
        def __init__(self, prm):
            started.append(prm)

        def start(self):
            pass

    import sunback.run.run_server_github as mod
    orig = (mod.Parameters, mod.SingleRunner)
    try:
        mod.Parameters, mod.SingleRunner = (lambda: p), RecordingRunner
        mod.run_server_github()
    finally:
        mod.Parameters, mod.SingleRunner = orig
    p.is_debug.assert_called_once_with(True)
