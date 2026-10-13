# Test fixtures

| File | What | Source |
|---|---|---|
| `manifest_171.json` | One manifest fragment, byte for byte as served | Public GET of `https://the-sun-now.s3.us-east-2.amazonaws.com/manifest/171.json` on 2026-10-13T03:49:45Z; 362 bytes; ETag `09045363c83c45cd6654d799ec09e4a5`; Last-Modified 2026-10-13 03:43:09 GMT; sha256 `5f1267d4fc548b1795f2f8c8ccff26ddb28ff73fcbcf453f613972a608042edc` |
| `manifest_index.json` | The product index, byte for byte as served (12 products) | Public GET of `https://the-sun-now.s3.us-east-2.amazonaws.com/manifest/index.json` on 2026-10-13T03:49:45Z; 4542 bytes; ETag `704c5a063a68cb961ec7b894c86bce43`; Last-Modified 2026-10-13 03:43:12 GMT; sha256 `4b2f1c25b789abf92aa093d11d3e21d969df6f0b5318fa82c5d84a51e01db1e9` |
| `make_fits.py` | Synthetic AIA-like FITS writer; every number in it is synthetic | Code (SB-4) |

The two manifests replace the SB-4 placeholders, which carried a `_fixture_note`
key saying they were built offline. A captured file carries no note: the
validator (`manifest.validate_fragment`) rejects unknown keys, and the contract
tests in `test_manifest_contract.py` skip any file that still says PLACEHOLDER.
Provenance therefore lives in this table. To recapture (read-only GETs; do not
edit or pretty-print the result), run the two `curl -fsS ... -o` commands in
SB-4 Task 2 Step 3 and update the rows above.
