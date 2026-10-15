# SB-17 handoff to the Website register: one-click handoffs from each AIA lightbox

What sunback publishes once SB-17 is merged and the Lambda is redeployed (Gilly's steps). Keys and
fields are in `aws_lambda/video_builder/CONTRACT.md`.

## Data

- Fragment fields, AIA singles only: `fits` (`fits/aia_<id>_latest.fits.gz`) and `fits_bytes` (its size).
  Both are absent until the reducer has written the file; then hide the FITS buttons.
- Rainbow, composite_uv and dem never carry them: they have no single source file.
- The file is the integrated frame behind the still, gzipped FITS with its full header. It is
  rewritten every run (latest only).

## Three buttons in the lightbox footer, keyboard reachable

1. **Download FITS (<size>)**: a plain link to `fits` with the `download` attribute. The size comes from
   `fits_bytes` (for example "4.1 MB"; computed on the page, never typed in).
2. **Print this Sun**: `https://myheliograph.com/store/?d=YYYY-MM-DD&wl=<id>`. These parameters already
   exist in the store (`_shareParams` in `api/solar-archive.js`: `d`, `wl`, optional `t` as `HH:MM`).
   The date is the still's UTC date (`obs_end`, else `updated`), clamped to at most the date from
   `https://myheliograph.com/api/data_frontier`. The store drops a date outside its range silently,
   so the clamp is required. If the frontier cannot be read, link the store without `d`.
   Never link heliograph.com.
3. **Open in HelioFITS Studio**: the same download, plus a one-line tip: "Then open it in HelioFITS
   Studio with File > Open." Studio has no open-from-URL yet; that is a request for the HelioFITS
   Studio register.

## Credit

Beside the FITS button, visible on phones too: "Imagery courtesy of NASA/SDO and the AIA science team."
(Gilly's decided line, 2026-10-12.)
