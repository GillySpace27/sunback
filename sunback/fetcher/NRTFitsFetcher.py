"""Fetch the N most-recent synoptic-NRT frames per wavelength and time-integrate.

Unlike ``WebFitsFetcher`` (which grabs a single ``mostrecent/`` frame), this reads
the timestamped ``H<HH>00/`` hour buckets so multiple recent frames are available,
then collapses them per wavelength into one ``AIAsynoptic<wave>.fits`` — a drop-in
for the existing RHEF/Upsilon/RainbowRGB pipeline, but cleaner (cosmic rays/transients
removed by the default median).

Pure logic lives in (and is unit-tested via) ``nrt_listing`` and ``nrt_integrate``;
this class is the network/orchestration shell, verified by a live ``workflow_dispatch``.
"""
import os
import shutil
import urllib.request
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from sunback.fetcher.WebFitsFetcher import WebFitsFetcher
from sunback.fetcher.nrt_listing import nrt_hour_dirs, select_recent_frames
from sunback.fetcher.nrt_integrate import write_integrated_synoptic

# --- Tunables ---------------------------------------------------------------
INTEGRATION_FRAMES = 3            # N frames to integrate (~9 min at 3-min cadence)
INTEGRATION_METHOD = "median"     # 'median' (cosmic-ray robust) | 'mean' | 'sum'
LOOKBACK_HOURS = 1                # also scan the previous hour bucket near the edge
# 7 EUV channels that get their own card on the page.
SERVE_WAVES = ["0171", "0193", "0211", "0304", "0335", "0094", "0131"]
# The rainbow composite's rgb3 channel needs 1600/1700 too, so we fetch+integrate
# them even though they are not served as standalone cards.
COMPOSITE_ONLY_WAVES = ["1600", "1700"]
FETCH_WAVES = SERVE_WAVES + COMPOSITE_ONLY_WAVES
# ----------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class NRTFitsFetcher(WebFitsFetcher):
    base_url = "https://jsoc1.stanford.edu/data/aia/synoptic/nrt/"
    description = "Get + time-integrate recent NRT Fits per wavelength"
    filt_name = "NRTFitsFetcher"

    # allow params to override the module defaults if present
    def _cfg(self, name, default):
        return getattr(self.params, name.lower(), None) or default

    def _now(self):
        return datetime.now(timezone.utc)

    def _list_dir_filenames(self, url):
        """Return the .fits filenames in an NRT hour-bucket directory (or [])."""
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8")
        except Exception as e:  # missing bucket / transient error -> skip
            print(f"NRTFitsFetcher: could not list {url}: {e}")
            return []
        soup = BeautifulSoup(html, "html.parser")
        return [
            a["href"] for a in soup.find_all("a")
            if a.get("href", "").endswith(".fits")
        ]

    def fetch_fits_files(self):
        """Override: gather N recent frames per wave, integrate to synoptic FITS."""
        if not self.params.get_fits:
            return self.params.local_fits_paths()

        n = int(self._cfg("integration_frames", INTEGRATION_FRAMES))
        method = self._cfg("integration_method", INTEGRATION_METHOD)
        fits_dir = self.params.fits_directory()
        temp_dir = os.path.join(self.params.temp_directory(), "nrt_frames")
        os.makedirs(fits_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        if self.destroy:
            self.delete_directory_items(self.fits_folder)

        # 1. list recent hour buckets, accumulate filenames + their dir URLs
        listing = {}          # filename -> full url
        for dir_url in nrt_hour_dirs(self._now(), self.base_url, LOOKBACK_HOURS):
            for name in self._list_dir_filenames(dir_url):
                listing.setdefault(name, dir_url + name)

        # 2. pick the N newest per wavelength (incl. composite-only 1600/1700)
        selected = select_recent_frames(list(listing), FETCH_WAVES, n)

        out_paths = []
        for wave, names in selected.items():
            local_frames = []
            for name in names:
                local = os.path.join(temp_dir, name)
                if self.download_url(listing[name], local):
                    local_frames.append(local)
            if not local_frames:
                print(f"NRTFitsFetcher: no frames downloaded for {wave}")
                continue
            # 3. integrate -> AIAsynoptic<wave>.fits (drop-in for the pipeline)
            out = os.path.join(fits_dir, f"AIAsynoptic{wave}.fits")
            write_integrated_synoptic(local_frames, out, method=method)
            out_paths.append(out)

        if self.params.destroy:
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f" ^  Integrated {len(out_paths)} wavelengths "
              f"({n}x {method}) from NRT\n", flush=True)
        return out_paths
