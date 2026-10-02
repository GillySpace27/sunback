"""Typed settings for the NRT reducer, read from environment variables (SB-5).

Defaults reproduce production exactly: bucket ``the-sun-now``, no key prefix,
data under ``sunback_data/renders`` relative to the working directory, median
of 5 frames, and ``image_times_readable.txt`` still written.

A staging run sets ``SUNBACK_PREFIX=staging/`` so every key the reducer writes
lands under ``staging/``. The video Lambda's S3 trigger matches prefix ``1k/``
only (``aws_lambda/video_builder/deploy.py:136-137``), so staging stills never
fire it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

DEFAULT_BUCKET = "the-sun-now"
INTEGRATION_METHODS = ("median", "mean", "sum")
_FALSE_WORDS = ("0", "false", "no")
_PREFIX_RE = re.compile(r"^(?:[A-Za-z0-9_-][A-Za-z0-9._-]*/)*$")


@dataclass(frozen=True)
class NrtSettings:
    bucket: str = DEFAULT_BUCKET        # SUNBACK_BUCKET
    prefix: str = ""                    # SUNBACK_PREFIX; "" or ends with "/", never starts with "/"
    data_dir: str | None = None         # SUNBACK_DATA_DIR; None keeps "sunback_data/renders"
    integration_frames: int = 5         # SUNBACK_INTEGRATION_FRAMES
    integration_method: str = "median"  # SUNBACK_INTEGRATION_METHOD
    write_readable_times: bool = True   # SUNBACK_WRITE_READABLE_TIMES ("0", "false", "no" -> False)

    def __post_init__(self):
        if not _PREFIX_RE.match(self.prefix):
            raise ValueError(
                f"SUNBACK_PREFIX must be empty or like 'staging/' "
                f"(segments of letters, digits, '.', '_', '-', each ending in '/'); got {self.prefix!r}"
            )
        if self.integration_method not in INTEGRATION_METHODS:
            raise ValueError(
                f"SUNBACK_INTEGRATION_METHOD must be one of {INTEGRATION_METHODS}; "
                f"got {self.integration_method!r}"
            )
        if self.integration_frames < 1:
            raise ValueError(f"SUNBACK_INTEGRATION_FRAMES must be >= 1; got {self.integration_frames}")
        if not self.bucket:
            raise ValueError("SUNBACK_BUCKET must not be empty")

    @classmethod
    def from_env(cls, environ=None) -> "NrtSettings":
        env = os.environ if environ is None else environ
        readable = env.get("SUNBACK_WRITE_READABLE_TIMES", "1").strip().lower()
        return cls(
            bucket=env.get("SUNBACK_BUCKET") or DEFAULT_BUCKET,
            prefix=env.get("SUNBACK_PREFIX", ""),
            data_dir=env.get("SUNBACK_DATA_DIR") or None,
            integration_frames=int(env.get("SUNBACK_INTEGRATION_FRAMES") or 5),
            integration_method=env.get("SUNBACK_INTEGRATION_METHOD") or "median",
            write_readable_times=readable not in _FALSE_WORDS,
        )

    def prefixed(self, key: str) -> str:
        return self.prefix + key
