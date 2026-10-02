"""SB-5 production-safety rails: today's keys by default, staging/ on request,
no bucket wipe, no unflagged sunback-serve.

Every S3 call goes to a stand-in; nothing here reaches AWS.
"""
import importlib
import os
import sys
from unittest.mock import MagicMock

import boto3
import numpy as np
import pytest

aws = importlib.import_module("sunback.putter.AwsPutter")


class FakeS3Client:
    """Records upload_file; any other S3 call fails the test."""

    def __init__(self):
        self.uploads = []

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None, **kwargs):
        assert os.path.exists(Filename), Filename
        self.uploads.append((Bucket, Key, dict(ExtraArgs or {})))

    def __getattr__(self, name):
        raise AssertionError(f"unexpected S3 call: {name}")


class StandInBucket:
    """Takes the place of the pre-SB-5 module-level bucket; records a wipe instead of doing it."""

    def __init__(self):
        self.wiped = 0

    @property
    def objects(self):
        return self

    def all(self):
        return self

    def delete(self):
        self.wiped += 1
        return []


def test_empty_the_bucket_always_raises(monkeypatch):
    stand_in = StandInBucket()
    monkeypatch.setattr(aws, "bucket", stand_in, raising=False)
    monkeypatch.setattr(aws, "_S3_CLIENT", FakeS3Client(), raising=False)
    putter = aws.AwsPutter.__new__(aws.AwsPutter)
    with pytest.raises(RuntimeError, match="empty_the_bucket is fenced"):
        putter.empty_the_bucket()
    assert stand_in.wiped == 0
