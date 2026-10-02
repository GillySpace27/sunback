"""Runner retry behaviour (SB-11).

Run mode (``is_debug()`` False: the wallpaper client) retries a failed batch
up to ``fail_max`` times; debug mode (the production reducer,
``run_server_github(debug=True)``) raises on the first failure. Production
keeps raising on purpose: a retried reducer run could publish a half-failed
wave, and whether production retries is Gilly's call (overview Q14).
"""
from types import SimpleNamespace

import pytest

from sunback.run.run import Runner


class FlakyRunner(Runner):
    def __init__(self, params, failures):
        super().__init__(params)
        self.failures = failures
        self.calls = 0

    def process(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"failure {self.calls}")


def params(debug):
    return SimpleNamespace(is_debug=lambda: debug, stop_after_one=lambda: True)


def test_run_mode_retries_after_one_failure():
    runner = FlakyRunner(params(debug=False), failures=1)
    runner.start(verb=False)
    assert runner.calls == 2


def test_run_mode_gives_up_after_ten_failures():
    runner = FlakyRunner(params(debug=False), failures=50)
    with pytest.raises(SystemExit) as exc:
        runner.start(verb=False)
    assert exc.value.code == 1
    assert runner.calls == 10


def test_debug_mode_raises_on_first_failure():
    runner = FlakyRunner(params(debug=True), failures=1)
    with pytest.raises(RuntimeError, match="failure 1"):
        runner.start(verb=False)
    assert runner.calls == 1
