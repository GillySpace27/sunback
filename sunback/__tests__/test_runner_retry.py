"""Runner failure behaviour (SB-11, restored by the review).

Both run mode (``is_debug()`` False: the wallpaper client and sunback-serve) and
debug mode (the production reducer, ``run_server_github(debug=True)``) re-raise
the first failure of a batch, as they always did. The ``fail_max`` retry loop
below the ``raise`` in ``Runner.__run_mode`` is therefore unreachable. SB-11
had removed the ``raise`` so run mode would retry up to 10 times with no pause
and then exit 1; that changes sunback-serve and the client, so whether to make
it so is Gilly's call (overview Q14, open). These tests pin today's behaviour.
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


def test_run_mode_raises_on_first_failure():
    runner = FlakyRunner(params(debug=False), failures=1)
    with pytest.raises(RuntimeError, match="failure 1"):
        runner.start(verb=False)
    assert runner.calls == 1


def test_run_mode_does_not_retry_a_persistent_failure():
    runner = FlakyRunner(params(debug=False), failures=50)
    with pytest.raises(RuntimeError, match="failure 1"):
        runner.start(verb=False)
    assert runner.calls == 1


def test_debug_mode_raises_on_first_failure():
    runner = FlakyRunner(params(debug=True), failures=1)
    with pytest.raises(RuntimeError, match="failure 1"):
        runner.start(verb=False)
    assert runner.calls == 1
