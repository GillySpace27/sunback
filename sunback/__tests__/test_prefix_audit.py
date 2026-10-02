"""SB-prefix: offline audit that SUNBACK_PREFIX reaches EVERY key the reducer writes.

Decision (Gilly, 2026-10-02): wire the input, add an offline audit, do not dispatch.
A staging run that wrote even one production key would overwrite what the public page
shows, so this file proves two things without touching AWS:

1. Drive the reducer's upload code (AwsPutter.put, the only S3 writer on the path of
   sunback/run/run_server_github.py) against a client that records EVERY call, not just
   upload_file. With SUNBACK_PREFIX=staging/ every key written starts with "staging/";
   with it unset (or empty, which is what the workflow passes on push and schedule) the
   keys are exactly the production keys the older tests pin (TODAYS_KEYS).
2. A static guard: every S3 write call site (and every boto3 user) under sunback/ and
   aws_lambda/ is on the audit list below. A new write call site fails the test until
   someone adds it here and says which test covers it.

Nothing here reaches AWS: boto3.client returns the recorder and boto3.resource refuses.
"""
import ast
import importlib
import inspect
import re
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import boto3
import cv2
import numpy as np
import pytest

from sunback.__tests__.test_aws_putter_guards import LOCAL_PNGS, TODAYS_KEYS
from sunback.__tests__.test_awsputter import _sb9_integrated

aws = importlib.import_module("sunback.putter.AwsPutter")

ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_KEYS = [k for _, k, _, _ in TODAYS_KEYS]
STAGING = ["staging/", "staging/run-42/"]


# --- dynamic audit: a client that records every call -----------------------------------------------
READ_ONLY = {"get_object", "head_object", "list_objects", "list_objects_v2", "download_file",
             "download_fileobj", "get_object_acl", "head_bucket", "get_paginator"}


class RecordingS3:
    """Records every method call; anything not in READ_ONLY counts as a write."""

    def __init__(self):
        self.writes = []   # (method, bucket, key)
        self.reads = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def call(*args, **kwargs):
            if name in READ_ONLY:
                self.reads.append(name)
                return {}
            for bucket, key in self._targets(name, args, kwargs):
                self.writes.append((name, bucket, key))
            return {}
        return call

    @staticmethod
    def _targets(name, args, kwargs):
        """(bucket, key) pairs one write call touches; fails the test when it cannot tell."""
        if name == "delete_objects":
            bucket = kwargs.get("Bucket")
            return [(bucket, o["Key"]) for o in kwargs["Delete"]["Objects"]]
        if name in ("upload_file", "upload_fileobj", "copy"):   # (Filename|CopySource, Bucket, Key)
            bucket = kwargs.get("Bucket", args[1] if len(args) > 1 else None)
            key = kwargs.get("Key", args[2] if len(args) > 2 else None)
        else:                                                    # put_object, copy_object, ...
            bucket, key = kwargs.get("Bucket"), kwargs.get("Key")
        assert key is not None, f"unauditable S3 write {name}{args!r} {sorted(kwargs)}"
        return [(bucket, key)]


@pytest.fixture
def s3(monkeypatch):
    """Every way of getting an S3 client or resource lands on the recorder (or refuses)."""
    rec = RecordingS3()
    monkeypatch.setattr(aws, "_S3_CLIENT", rec)

    def client(service, *a, **k):
        assert service == "s3", f"unexpected boto3 client {service!r}"
        return rec

    def resource(*a, **k):
        raise AssertionError("boto3.resource used: its write calls (Object.put, Bucket.upload_file) are not audited")

    monkeypatch.setattr(boto3, "client", client)
    monkeypatch.setattr(boto3, "resource", resource)
    for name in ("SUNBACK_BUCKET", "SUNBACK_PREFIX", "SUNBACK_WRITE_READABLE_TIMES"):
        monkeypatch.delenv(name, raising=False)
    return rec


def drive_reducer_upload(tmp_path, with_fits):
    """AwsPutter.put() over every served product, the DEM scan video and the time files."""
    pngs = []
    for name in LOCAL_PNGS:
        path = tmp_path / name
        cv2.imwrite(str(path), np.zeros((16, 16), np.uint8))
        pngs.append(str(path))
    (tmp_path / "dem").mkdir()
    (tmp_path / "dem" / "a_temp_video_small.mp4").write_bytes(b"mp4")
    fits_dir = tmp_path / "fits"
    if with_fits:   # header times found: exercises the metadata, tEXt re-save and sidecar branch
        _sb9_integrated(fits_dir, "0171", ["2026-09-28T11:52:00Z", "2026-09-28T12:00:00Z"])

    params = MagicMock()
    params.multi_pool = None
    params.local_imgs_paths.return_value = pngs
    params.imgs_top_directory.return_value = str(tmp_path)
    params.base_directory.return_value = str(tmp_path)
    params.time_path.return_value = str(tmp_path / "image_times.txt")
    params.local_fits_paths.return_value = [str(tmp_path / "AIAsynoptic0171.fits")]
    params.fits_directory.return_value = str(fits_dir)

    putter = aws.AwsPutter.__new__(aws.AwsPutter)  # skip Processor.__init__ (no FITS pipeline needed)
    putter.params = params
    putter.ii = 0
    putter.pbar = None
    putter.to_upload = None
    putter.load_this_fits_frame = lambda fits_path=None, in_name=None, quiet=False: (
        None, "0171", "2026-09-28T12:00:00.00", None, None, None)
    putter.clean_time_string = lambda t, zone=None, out_fmt=None: f"{zone}|{t}"
    putter.put()


@pytest.mark.parametrize("with_fits", [False, True])
@pytest.mark.parametrize("prefix", STAGING)
def test_staging_prefix_on_every_written_key(s3, tmp_path, monkeypatch, prefix, with_fits):
    monkeypatch.setenv("SUNBACK_PREFIX", prefix)
    drive_reducer_upload(tmp_path, with_fits)
    assert s3.writes, "the audit drove no write at all"
    stray = [(m, k) for m, _, k in s3.writes if not k.startswith(prefix)]
    assert not stray, f"keys written outside {prefix!r} (a staging run would hit production): {stray}"
    assert [k for _, _, k in s3.writes] == [prefix + k for k in PRODUCTION_KEYS]
    assert {m for m, _, _ in s3.writes} == {"upload_file"}, "a new kind of S3 write joined the reducer path"
    assert {b for _, b, _ in s3.writes} == {"the-sun-now"}


@pytest.mark.parametrize("with_fits", [False, True])
@pytest.mark.parametrize("value", [None, ""])
def test_unset_or_empty_prefix_writes_exactly_the_production_keys(s3, tmp_path, monkeypatch, value, with_fits):
    """The workflow passes SUNBACK_PREFIX='' (not unset) on push and schedule: both are production."""
    if value is not None:
        monkeypatch.setenv("SUNBACK_PREFIX", value)
    drive_reducer_upload(tmp_path, with_fits)
    assert [k for _, _, k in s3.writes] == PRODUCTION_KEYS
    assert {b for _, b, _ in s3.writes} == {"the-sun-now"}


def test_every_staging_key_is_a_production_key_under_the_prefix(s3, tmp_path, monkeypatch):
    """No staging-only and no production-only key: the two runs differ by the prefix alone."""
    monkeypatch.setenv("SUNBACK_PREFIX", "staging/")
    drive_reducer_upload(tmp_path, True)
    assert sorted(k[len("staging/"):] for _, _, k in s3.writes) == sorted(PRODUCTION_KEYS)


def test_run_server_github_path_has_one_s3_writer(monkeypatch):
    """The putters/processors/fetchers run_server_github wires: only AwsPutter's module writes to S3."""
    rsg = importlib.import_module("sunback.run.run_server_github")
    started = []

    class RecordingRunner:
        def __init__(self, params):
            self.params = params

        def start(self):
            started.append(self.params)

    monkeypatch.setattr(rsg, "SingleRunner", RecordingRunner)
    monkeypatch.setenv("SUNBACK_PREFIX", "staging/")
    rsg.run_server_github()
    assert len(started) == 1
    p = started[0]
    classes = list(p._fetchers) + list(p._processors) + list(p._putters)
    writer_files = {rel for rel in scan_tree(ROOT) if rel.startswith("sunback/")}
    touching = set()
    for cls in classes:
        rel = Path(inspect.getsourcefile(cls)).resolve().relative_to(ROOT).as_posix()
        if rel in writer_files:
            touching.add(rel)
    assert touching == {"sunback/putter/AwsPutter.py"}, touching
    assert aws.AwsPutter in p._putters


# --- static guard: every S3 write call site is on the audit list ------------------------------------
WRITE_ATTRS = {
    "put_object", "upload_file", "upload_fileobj", "copy_object", "delete_object", "delete_objects",
    "put_object_acl", "put_object_tagging", "create_multipart_upload", "upload_part", "upload_part_copy",
    "complete_multipart_upload", "create_bucket", "delete_bucket", "put_bucket_policy",
    "put_bucket_notification_configuration", "put_bucket_lifecycle_configuration", "put_bucket_cors",
    "put_public_access_block",
}
# Generic names are flagged only when the receiver looks like an S3 handle (resource API).
AMBIGUOUS_ATTRS = {"put", "copy", "delete", "upload"}
S3_RECEIVER = re.compile(r"s3|bucket|\.Object\(|objects", re.IGNORECASE)
CLI_WRITE = re.compile(r"aws\s+s3|s3api|\bs3\s+(?:cp|sync|mv|rm)\b")
SCAN_ROOTS = ("sunback", "aws_lambda")
SKIP_DIRS = {"__tests__", "__pycache__", "build", "attic"}

# path -> {call: count}. "boto3" counts `import boto3` / `from boto3 ...` lines (a new boto3 user
# is a new place that might write). Every entry names the test or reason that covers it.
AUDITED = {
    # The reducer's only S3 writer: upload_public() -> get_s3_client().upload_file(settings.prefixed(key)).
    # Covered by test_staging_prefix_on_every_written_key and test_unset_or_empty_prefix_*.
    "sunback/putter/AwsPutter.py": {"upload_file": 1, "boto3": 1},
    # Read only (download_file, objects.filter): the wallpaper client, not the reducer.
    "sunback/fetcher/AwsImgFetcher.py": {"boto3": 1},
    # Legacy movie Lambda for another bucket (gillyspace27-test-billboard); imported by nothing in
    # the package and not on the run_server_github path (test_run_server_github_path_has_one_s3_writer).
    "sunback/movie/lambda_function_movie.py": {"upload_file": 4, "boto3": 1},
    # The video Lambda: runs in AWS on a 1k/ trigger, writes under the trigger key's prefix.
    # Covered by test_handler_isolation.py (staging keys stay under staging/).
    "aws_lambda/video_builder/handler.py": {"put_object": 3, "copy_object": 2, "delete_object": 1,
                                            "upload_file": 2, "boto3": 1},
    # Manual deploy tools Gilly runs on his Mac against his account (AGENTS.md: never run by agents).
    "aws_lambda/video_builder/deploy.py": {"put_object": 1, "put_bucket_notification_configuration": 1,
                                           "boto3": 1},
    "aws_lambda/video_builder/layer/build_layer.py": {"put_object": 1, "boto3": 2},
    "aws_lambda/video_builder/deploy_code.py": {"boto3": 1},
}


def scan_source(text):
    """{call: count} of S3 write call sites and boto3 imports in one Python source."""
    found = {}

    def add(name):
        found[name] = found.get(name, 0) + 1

    for line in text.splitlines():
        if CLI_WRITE.search(line) and not line.lstrip().startswith("#"):
            add("aws-cli")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # old files with invalid escape sequences still parse
            tree = ast.parse(text)
    except SyntaxError:
        add("unparseable")
        return found
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "boto3" for a in node.names):
            add("boto3")
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "boto3":
            add("boto3")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in WRITE_ATTRS:
                add(attr)
            elif attr in AMBIGUOUS_ATTRS and S3_RECEIVER.search(ast.unparse(node.func.value)):
                add(attr)
    return found


def scan_tree(root):
    """{relative posix path: {call: count}} for every Python file under the scan roots."""
    out = {}
    for top in SCAN_ROOTS:
        for path in sorted((Path(root) / top).rglob("*.py")):
            rel_parts = path.relative_to(root).parts
            if SKIP_DIRS & set(rel_parts):
                continue
            found = scan_source(path.read_text(encoding="utf-8", errors="replace"))
            if found:
                out[path.relative_to(root).as_posix()] = found
    return out


def test_every_s3_write_call_site_is_on_the_audit_list():
    found = scan_tree(ROOT)
    unaudited = {p: c for p, c in found.items() if c != AUDITED.get(p)}
    stale = {p: c for p, c in AUDITED.items() if p not in found}
    assert not unaudited, (
        "S3 write call sites or boto3 users that are not on the SB-prefix audit list. Make sure each "
        "write honours NrtSettings.prefix (or is not on the reducer path), add a test that proves it, "
        f"then list it in AUDITED: {unaudited}")
    assert not stale, f"audit entries with no matching call site (update AUDITED): {stale}"


def test_the_scanner_can_fail():
    """The guard is only worth something if a new write site changes what it reports."""
    assert scan_source("s3.put_object(Bucket=b, Key=k, Body=x)") == {"put_object": 1}
    assert scan_source("import boto3\nboto3.client('s3').upload_file(a, b, c)") == {"boto3": 1, "upload_file": 1}
    assert scan_source("bucket.objects.all().delete()") == {"delete": 1}
    assert scan_source("s3.Object(b, k).put(Body=x)") == {"put": 1}
    assert scan_source("os.system('aws s3 cp a s3://b/c')") == {"aws-cli": 1}
    assert scan_source("d = {}.copy()\nputter.put()") == {}
    assert scan_source("def broken(:") == {"unparseable": 1}


def test_a_new_write_file_is_flagged(tmp_path):
    (tmp_path / "sunback" / "putter").mkdir(parents=True)
    (tmp_path / "aws_lambda").mkdir()
    (tmp_path / "sunback" / "putter" / "NewPutter.py").write_text(
        "def go(client):\n    client.put_object(Bucket='the-sun-now', Key='x', Body=b'')\n")
    assert scan_tree(tmp_path) == {"sunback/putter/NewPutter.py": {"put_object": 1}}
    assert "sunback/putter/NewPutter.py" not in AUDITED
