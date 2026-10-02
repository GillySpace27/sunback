# Changelog

Versions are bare tags (`0.6.17.3`), not `v*`. The entries below 0.6.17.4 were
reconstructed by SB-12 from git tags, tag commit subjects and PyPI upload dates
(read 2026-10-02); nothing earlier than 0.6.17.1 is described.

## Unreleased

- Packaging: `pyproject.toml` is the only metadata source; `sunback.__version__`
  comes from the installed distribution; the wheel holds only `sunback/` (88
  files instead of 207 on the builds of 2026-10-02); Python 3.11 or newer; `numpy`
  and `tqdm` declared; the IDL colour table ships inside the package. (SB-12)

## 0.6.17.3 (tagged 2026-07-09, on PyPI 2026-07-10)

- DesktopPutter: set the wallpaper through the native NSWorkspace API (fixes the
  no-op for background LaunchAgents). The good client release.

## 0.6.17.2 (tagged 2026-07-09, on PyPI 2026-07-10)

- DesktopPutter: retry and verify the macOS wallpaper set (fixes a silent no-op on
  macOS 14 and newer). Superseded by 0.6.17.3.

## 0.6.17.1 (tagged 2026-07-09, on PyPI 2026-07-10)

- Fetchers: point the scraper at the new `1k/` S3 prefix. Superseded by 0.6.17.3.

## Older tags

- `v1.0.0`: annotated 2025-01-31 ("Release v1.0.0") on commit 3b6473e of
  2025-01-27. Not on PyPI. What it means relative to 0.6.x is Gilly's to say.
- `v0.2.0`: on commit ae038be of 2025-01-23. Not on PyPI.
- Tags `0.0.1` to `0.0.106` (2019-2020) and `0.1.0` to `0.1.4` (not every number) exist on GitHub.
- PyPI also carries 0.1.3 to 0.6.17 (2021-01-28 to 2025-01-14); those versions after
  0.1.4 have no matching tag in this repository.
