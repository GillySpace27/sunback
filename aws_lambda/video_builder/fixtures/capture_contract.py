"""Capture and check golden fixtures for the public imagery contract (suite SU-9).

Run from the sunback root:

  python3 -m aws_lambda.video_builder.fixtures.capture_contract capture [--version N] [--out DIR]
                                                   [--base-url URL] [--synthesize-index] [--offline]
  python3 -m aws_lambda.video_builder.fixtures.capture_contract verify DIR
  python3 -m aws_lambda.video_builder.fixtures.capture_contract same DIR_A DIR_B

capture makes read-only GET requests to the public bucket (manifest/index.json, manifest/171.json,
manifest/rainbow.json, image_times.txt), validates what came back against the schemas in ../schema/,
and writes fixtures/contract-v<N>/ only if everything validates. It never overwrites a fixture:
versions are kept forever, a key change makes contract-v<N+1>. When manifest/index.json is not
served yet (the Lambda that writes it is not deployed), --synthesize-index builds it from the twelve
live fragments with manifest.build_index and says so in README.md.
--offline serves the same four documents from manifest.py with fixed times instead of reading a bucket
(no network at all); the README then says the fixture was NOT captured from the live bucket.
verify checks a fixture folder against its SHA256SUMS and the schemas. same compares two folders
byte for byte (producer copy against a consumer copy).
Exit codes: capture 0 written, 1 refused (nothing written), 2 destination exists; verify and same
0 or 1. Standard library only.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

from aws_lambda.video_builder.manifest import (PRODUCTS, build_index, build_manifest_fragment, versioned_still_key,
                                               versioned_video_key)
from aws_lambda.video_builder.schema.validate import validate

HERE = pathlib.Path(__file__).resolve().parent
SCHEMA_DIR = HERE.parent / "schema"
DEFAULT_BASE_URL = "https://the-sun-now.s3.us-east-2.amazonaws.com/"
OFFLINE_BASE_URL = "offline://manifest.py/"
FILES = ("index.json", "fragment-171.json", "fragment-rainbow.json", "image_times.txt", "README.md")
TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?\s*$")


def urllib_fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "sunback-contract-capture", "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, b""
    except (urllib.error.URLError, OSError):
        return 0, b""


def offline_fetch():
    """A fetch function that serves the four documents built from manifest.py, with fixed times.

    Same shape as urllib_fetch: fetch(url) -> (status, body). Every product carries the versioned keys and the
    provenance window, so the fixture exercises every optional field a reader may meet today.
    """
    generated, stamp = "2026-10-02T12:00:00Z", "20261002T120000"
    fragments = [build_manifest_fragment(
        p["id"], updated=generated, frame_count=144, integration={"frames": 5, "method": "median"},
        video_v=versioned_video_key(p["id"], stamp), still_v=versioned_still_key(p["id"], stamp),
        through=generated, obs_start="2026-10-02T11:50:00Z", obs_end="2026-10-02T11:58:00Z") for p in PRODUCTS]
    docs = {f"manifest/{f['id']}.json": (json.dumps(f, indent=1) + "\n").encode("utf-8") for f in fragments}
    docs["manifest/index.json"] = (json.dumps(build_index(fragments, generated), indent=1) + "\n").encode("utf-8")
    docs["image_times.txt"] = b"2026-10-02T11:58:02.081\n"

    def fetch(url):
        key = url[len(OFFLINE_BASE_URL):]
        return (200, docs[key]) if key in docs else (404, b"")
    return fetch


def _schema(name):
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def check_documents(docs):
    """Error strings for the captured documents (README.md is not checked); an empty list means all valid."""
    errors = []
    parsed = {}
    for name in ("index.json", "fragment-171.json", "fragment-rainbow.json"):
        try:
            parsed[name] = json.loads(docs[name].decode("utf-8"))
        except (KeyError, ValueError) as err:
            errors.append(f"{name}: not JSON: {err}")
    fragment_schema = _schema("manifest-fragment.schema.json")
    for name, pid in (("fragment-171.json", "171"), ("fragment-rainbow.json", "rainbow")):
        if name in parsed:
            errors.extend(f"{name}: {e}" for e in validate(parsed[name], fragment_schema))
            if isinstance(parsed[name], dict) and parsed[name].get("id") != pid:
                errors.append(f"{name}: id is {parsed[name].get('id')!r}, expected {pid!r}")
    if "index.json" in parsed:
        errors.extend(f"index.json: {e}" for e in validate(parsed["index.json"], _schema("index.schema.json")))
        products = parsed["index.json"].get("products") if isinstance(parsed["index.json"], dict) else None
        if isinstance(products, list):
            ids = [p.get("id") for p in products if isinstance(p, dict)]
            missing = [p["id"] for p in PRODUCTS if p["id"] not in ids]
            if missing:
                errors.append("index.json: products lack " + ", ".join(missing))
    if not TIME_RE.match(docs.get("image_times.txt", b"").decode("utf-8", "replace")):
        errors.append("image_times.txt: not a UTC time line")
    return errors


def _readme(version, now, index_note, base_url=DEFAULT_BASE_URL):
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if base_url == DEFAULT_BASE_URL:
        origin = f"Captured {stamp} from {base_url} by capture_contract.py (read-only GET requests).\n\n"
    else:
        origin = (f"Built {stamp} by capture_contract.py from {base_url}. **NOT captured from the live bucket**: a "
                  "stand-in served documents that manifest.py builds, so this fixture pins the format the producer "
                  "code writes, not bytes the live bucket served. A capture of the live bucket takes the next free "
                  "version number (versions are never overwritten).\n\n")
    return (
        f"# contract-v{version}\n\n"
        + origin +
        f"- index.json: {index_note}\n"
        "- fragment-171.json and fragment-rainbow.json: manifest/171.json and manifest/rainbow.json as served\n"
        "- image_times.txt: image_times.txt as served\n\n"
        f"Contract version {version} is the key layout and the fields described in "
        "aws_lambda/video_builder/CONTRACT.md on the capture date. This folder is kept forever: a change to a "
        f"required key or its meaning makes contract-v{version + 1}/ beside it. Byte-identical consumer copies: "
        f"Website tools/tests/fixtures/contract-v{version}/ and heliogram infra/contract/contract-v{version}/. "
        "Compare a copy with `python3 -m aws_lambda.video_builder.fixtures.capture_contract same <A> <B>`.\n"
    ).encode("utf-8")


def _sums(docs):
    return "".join(f"{hashlib.sha256(docs[n]).hexdigest()}  {n}\n" for n in sorted(FILES)).encode("utf-8")


def capture(fetch, out_dir, version=1, base_url=DEFAULT_BASE_URL, synthesize=False, now=None):
    now = now or datetime.now(timezone.utc)
    out = pathlib.Path(out_dir)
    if out.exists():
        print(f"REFUSING: {out} exists; fixture versions are kept forever, use a new version number")
        return 2
    docs = {}
    status, body = fetch(base_url + "manifest/index.json")
    index_note = "manifest/index.json as served"
    if status == 200:
        docs["index.json"] = body
    elif synthesize:
        fragments, missing = [], []
        for p in PRODUCTS:
            s, b = fetch(base_url + f"manifest/{p['id']}.json")
            try:
                if s != 200:
                    raise ValueError(f"HTTP {s}")
                fragments.append(json.loads(b.decode("utf-8")))
            except ValueError:
                missing.append(p["id"])
        if missing:
            print("CANNOT SYNTHESIZE: no usable fragment for " + ", ".join(missing))
            return 1
        generated = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        docs["index.json"] = (json.dumps(build_index(fragments, generated), indent=1) + "\n").encode("utf-8")
        index_note = (f"synthesized with manifest.build_index from the live fragments "
                      f"(manifest/index.json answered HTTP {status})")
    else:
        print(f"manifest/index.json answered HTTP {status}: the Lambda that writes it is not live. "
              "Re-run with --synthesize-index to build it from the fragments, or deploy first.")
        return 1
    for pid in ("171", "rainbow"):
        s, b = fetch(base_url + f"manifest/{pid}.json")
        if s != 200:
            print(f"manifest/{pid}.json answered HTTP {s}")
            return 1
        docs[f"fragment-{pid}.json"] = b
    s, b = fetch(base_url + "image_times.txt")
    if s != 200:
        print(f"image_times.txt answered HTTP {s}")
        return 1
    docs["image_times.txt"] = b
    errors = check_documents(docs)
    if errors:
        for e in errors:
            print(f"INVALID {e}")
        print(f"NOT WRITTEN: {len(errors)} problem(s); the fixture would not match the schema")
        return 1
    docs["README.md"] = _readme(version, now, index_note, base_url)
    out.mkdir(parents=True)
    for name, data in docs.items():
        (out / name).write_bytes(data)
    (out / "SHA256SUMS").write_bytes(_sums(docs))
    print(f"captured {out}: {len(docs)} files, {len(json.loads(docs['index.json'])['products'])} products")
    return 0


def verify(root):
    root = pathlib.Path(root)
    problems = []
    try:
        listed = {}
        for n, line in enumerate((root / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), 1):
            digest, sep, name = line.partition("  ")
            if not sep or len(digest) != 64:
                problems.append(f"SHA256SUMS:{n}: expected '<64 hex>  <name>'")
            else:
                listed[name] = digest
    except OSError:
        problems.append("MISSING SHA256SUMS")
        listed = {}
    docs = {}
    for name in FILES:
        try:
            docs[name] = (root / name).read_bytes()
        except OSError:
            problems.append(f"MISSING {name}")
            continue
        if name in listed and hashlib.sha256(docs[name]).hexdigest() != listed[name]:
            problems.append(f"MISMATCH {name}")
        if name not in listed:
            problems.append(f"UNLISTED {name}")
    if all(n in docs for n in FILES):
        problems.extend(f"INVALID {e}" for e in check_documents(docs))
    for p in problems:
        print(p)
    if problems:
        print(f"verify: {len(problems)} problem(s) in {root}")
        return 1
    print(f"OK {root}: {len(FILES)} files match SHA256SUMS and the schemas")
    return 0


def same(a, b):
    a, b = pathlib.Path(a), pathlib.Path(b)
    names = sorted({p.name for p in a.iterdir()} | {p.name for p in b.iterdir()})
    different = [n for n in names
                 if not (a / n).is_file() or not (b / n).is_file() or (a / n).read_bytes() != (b / n).read_bytes()]
    for n in different:
        print(f"DIFFERENT {n}")
    if different:
        print(f"same: {len(different)} file(s) differ between {a} and {b}")
        return 1
    print(f"IDENTICAL {a} {b}: {len(names)} files")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capture and check imagery contract fixtures.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--version", type=int, default=1)
    cap.add_argument("--out")
    cap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    cap.add_argument("--synthesize-index", action="store_true")
    cap.add_argument("--offline", action="store_true", help="build the documents from manifest.py; no network")
    ver = sub.add_parser("verify")
    ver.add_argument("dir")
    sm = sub.add_parser("same")
    sm.add_argument("a")
    sm.add_argument("b")
    args = ap.parse_args(argv)
    if args.cmd == "capture":
        out = args.out or str(HERE / f"contract-v{args.version}")
        base = args.base_url if args.base_url.endswith("/") else args.base_url + "/"
        if args.offline:
            return capture(offline_fetch(), out, args.version, OFFLINE_BASE_URL, args.synthesize_index)
        return capture(urllib_fetch, out, args.version, base, args.synthesize_index)
    if args.cmd == "verify":
        return verify(args.dir)
    return same(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
