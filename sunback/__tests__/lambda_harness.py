"""In-memory S3 and a fake ffmpeg for tests of aws_lambda/video_builder/handler.py.

API fixed by sunback/00-overview.md section 4 (producer SB-11; SB-8 creates this
file only if SB-11 has not landed). No network and no AWS: the handler's module
level boto3 client is replaced by FakeS3 after a reload.
"""
import importlib
import io
import json
import os
import stat
import sys
from datetime import datetime, timezone

from botocore.exceptions import ClientError


def _missing(operation, key):
    return ClientError({"Error": {"Code": "404", "Message": f"Not Found: {key}"}}, operation)


class FakeS3:
    """Dict-backed stand-in for the boto3 S3 client calls handler.py makes."""

    def __init__(self):
        self.objects = {}  # key -> {"Body": bytes, "Metadata": dict, "LastModified": datetime, **extra}
        self.calls = []    # (method name, key) in call order

    def _store(self, key, body, metadata=None, **extra):
        self.objects[key] = {"Body": body, "Metadata": dict(metadata or {}),
                             "LastModified": datetime.now(timezone.utc), **extra}

    def put_object(self, Bucket, Key, Body, Metadata=None, **extra):
        self.calls.append(("put_object", Key))
        self._store(Key, Body if isinstance(Body, bytes) else str(Body).encode("utf-8"), Metadata, **extra)
        return {}

    def get_object(self, Bucket, Key):
        self.calls.append(("get_object", Key))
        if Key not in self.objects:
            raise _missing("GetObject", Key)
        obj = self.objects[Key]
        return {"Body": io.BytesIO(obj["Body"]), "Metadata": dict(obj["Metadata"]),
                "LastModified": obj["LastModified"]}

    def head_object(self, Bucket, Key):
        self.calls.append(("head_object", Key))
        if Key not in self.objects:
            raise _missing("HeadObject", Key)
        obj = self.objects[Key]
        return {"Metadata": dict(obj["Metadata"]), "LastModified": obj["LastModified"],
                "ContentLength": len(obj["Body"])}

    def copy_object(self, Bucket, CopySource, Key, MetadataDirective="COPY", Metadata=None, **extra):
        self.calls.append(("copy_object", Key))
        src = CopySource["Key"]
        if src not in self.objects:
            raise _missing("CopyObject", src)
        meta = self.objects[src]["Metadata"] if MetadataDirective == "COPY" else (Metadata or {})
        self._store(Key, self.objects[src]["Body"], meta, **extra)
        return {}

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None, **kwargs):
        self.calls.append(("list_objects_v2", Prefix))
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        return {"Contents": [{"Key": k, "LastModified": self.objects[k]["LastModified"]} for k in keys],
                "IsTruncated": False}

    def delete_object(self, Bucket, Key):
        self.calls.append(("delete_object", Key))
        self.objects.pop(Key, None)
        return {}

    def download_file(self, Bucket, Key, Filename):
        self.calls.append(("download_file", Key))
        if Key not in self.objects:
            raise _missing("HeadObject", Key)
        with open(Filename, "wb") as fp:
            fp.write(self.objects[Key]["Body"])

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        self.calls.append(("upload_file", Key))
        extra = dict(ExtraArgs or {})
        meta = extra.pop("Metadata", None)
        with open(Filename, "rb") as fp:
            self._store(Key, fp.read(), meta, **extra)

    def json(self, key):
        """Decoded JSON body of one stored object (test convenience)."""
        return json.loads(self.objects[key]["Body"])


def fake_ffmpeg(tmp_path):
    """Write an executable that records its argv to argv.json and writes its last argv path."""
    path = os.path.join(str(tmp_path), "fake_ffmpeg")
    with open(path, "w") as fp:
        fp.write(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "here = os.path.dirname(os.path.abspath(__file__))\n"
            "with open(os.path.join(here, 'argv.json'), 'w') as fp:\n"
            "    json.dump(sys.argv, fp)\n"
            "with open(sys.argv[-1], 'wb') as fp:\n"
            "    fp.write(b'fake mp4')\n"
        )
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def load_handler(monkeypatch, fake_s3, ffmpeg_path, env=None):
    """Set the Lambda environment, reload handler.py and swap in the fake S3 client."""
    monkeypatch.setenv("FFMPEG_PATH", ffmpeg_path)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    import aws_lambda.video_builder.handler as handler
    handler = importlib.reload(handler)
    monkeypatch.setattr(handler, "s3", fake_s3)
    return handler
