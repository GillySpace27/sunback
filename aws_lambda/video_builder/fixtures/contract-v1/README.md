# contract-v1

Built 2026-10-02T18:28:29Z by capture_contract.py from offline://manifest.py/. **NOT captured from the live bucket**: a stand-in served documents that manifest.py builds, so this fixture pins the format the producer code writes, not bytes the live bucket served. A capture of the live bucket takes the next free version number (versions are never overwritten).

- index.json: manifest/index.json as served
- fragment-171.json and fragment-rainbow.json: manifest/171.json and manifest/rainbow.json as served
- image_times.txt: image_times.txt as served

Contract version 1 is the key layout and the fields described in aws_lambda/video_builder/CONTRACT.md on the capture date. This folder is kept forever: a change to a required key or its meaning makes contract-v2/ beside it. Byte-identical consumer copies: Website tools/tests/fixtures/contract-v1/ and heliogram infra/contract/contract-v1/. Compare a copy with `python3 -m aws_lambda.video_builder.fixtures.capture_contract same <A> <B>`.
