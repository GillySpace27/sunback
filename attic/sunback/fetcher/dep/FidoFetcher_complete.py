import os
import sys
import datetime
import logging
from time import strptime, mktime

import numpy as np
from astropy.io import fits
from tqdm import tqdm
from drms import DrmsExportError
from sunpy.net import Fido, attrs
from parfive import Downloader
import astropy.units as u
from sunpy.coordinates.sun import carrington_rotation_time

# Local module imports
from sunback.utils.time_util import (
    parse_time_string_to_local,
    define_time_range,
    define_recent_range,
)
from sunback.fetcher.Fetcher import Fetcher
from sunback.processor.SunPyProcessor import AIA_PREP_Processor

# Constants
DEFAULT_BASE_URL = "http://jsoc1.stanford.edu/data/aia/synoptic/mostrecent/"
JSOC_EMAIL = "chris.gilly@colorado.edu"

# Configure logging for verbose output
logging.basicConfig(level=logging.INFO)


class FidoFetcher(Fetcher):
    description = "Get Fits Files from the Internet using Fido"
    verbose = True
    filt_name = "Fido Fetcher"
    num_files_needed = None
    batch_id = 0
    needed_files = None
    results = None
    temp_folder = None
    fits_path = None

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.params = params or Parameters()  # Ensure params is never None
        self.SubDownloader = None
        self.reprocess_mode(rp)
        self.params.load_preset_time_settings()

    def fetch(self, params=None, quick=False, rp=None, verb=True):
        """Find the Most Recent Images"""
        # Ensure params is properly set and not None
        if params:
            self.params = params
        elif not self.params:
            raise ValueError("Parameters object is not initialized.")

        self.quick = quick
        self.rp = rp
        self.verb = verb

        # Ensure self.params.current_wave() is accessible before calling it
        if (
            not hasattr(self.params, "current_wave")
            or self.params.current_wave() is None
        ):
            raise AttributeError("The 'params' object is missing 'current_wave'.")

        self.fido_get_fits(self.params.current_wave(), temp=self.params.do_temp)

    def cleanup(self):
        """Cleanup resources after fetching files."""
        try:
            del self.fido_search_result
        except AttributeError:
            pass
        try:
            del self.needed_files
        except AttributeError:
            pass
        try:
            del self.results
        except AttributeError:
            pass
        super().cleanup()

    def setup_time_range(self):
        """Set up the time range for data retrieval, including Carrington rotations if specified."""
        if self.params.carrington() is not None:
            (
                self.params.carrington_start,
                self.params.carrington_end,
                self.params.num_frames,
            ) = self.params.carrington()

            times = self.carrington_to_time(
                self.params.carrington_start,
                self.params.carrington_end,
                self.params.num_frames,
            )

            # Debugging statements to track time range
            logging.info(f"Carrington start: {self.params.carrington_start}")
            logging.info(f"Carrington end: {self.params.carrington_end}")
            logging.info(f"Number of frames: {self.params.num_frames}")
            logging.info(f"Carrington start date: {times[0]}")
            logging.info(f"Carrington end date: {times[-1]}")
            logging.info(f"Generated times: {["------"+str(t)+"-----" for t in times]}")

            self.params.unpack_time_strings(times[0], times[-1])
            self.params.time_steps = times
        else:
            self.params.define_range()
            # Debug statement to show the time range when not using Carrington rotation
            logging.info(f"Start time after define_range: {self.params.start_time}")
            logging.info(f"End time after define_range: {self.params.end_time}")

    def carrington_to_time(self, carrington_start, carrington_end, num_frames):
        """
        Convert Carrington rotation numbers to times and generate intermediate steps.

        Args:
            carrington_start (float): Start Carrington rotation number.
            carrington_end (float): End Carrington rotation number.
            num_frames (int): Number of time steps (frames) between start and end rotations.

        Returns:
            list of datetime: List of times corresponding to the frames.
        """
        start_time = carrington_rotation_time(carrington_start).to_datetime()
        end_time = carrington_rotation_time(carrington_end).to_datetime()
        time_delta = (end_time - start_time) / num_frames
        if self.params.cadence_minutes is None:
            self.params.cadence_minutes = time_delta.to_datetime()

        return [start_time + i * time_delta for i in range(num_frames)]

    def fido_get_fits(self, current_wave, temp=False, num_frames=2):
        """Fetch FITS files using Fido."""
        self.load(self.params, wave=current_wave)
        have_file = self.determine_image_path() is not False
        need_file = self.params.download_files() and not have_file
        want_to_redo = self.reprocess_mode() and have_file

        if need_file or want_to_redo:
            self.print_load_banner(verb=self.verb)
            self.setup_time_range()
            self.fido_check_for_fits()
        else:
            prnt = self.params.n_fits if not self.params.do_single else "\b"
            logging.info(f"*\n^ Using {prnt} Cached Fits Files")

    def fido_check_for_fits(self, verb=None):
        """Find the science images."""
        from astropy import units as u

        if self.params.cadence_minutes() is None:
            pass

        self.verb = self.verb or verb
        logging.info(
            f"Looking for Images of {self.params.current_wave()} from \n{self.params.start_time_string} to"
            f"\n{self.params.end_time_string} with {self.params.cadence_minutes().to(u.day):0.3} days or "
            f"{self.params.cadence_minutes().to(u.s):0.3} seconds cadence..."
        )

        # Actual Fido search
        try:
            time_attr = attrs.Time(self.params.start_time, self.params.end_time)
            wave_attr = attrs.Wavelength(int(self.params.current_wave()) * u.angstrom)
            sample_attr = attrs.Sample(self.params.cadence_minutes())

            base_attrs = time_attr & wave_attr & sample_attr
            inst_attr = attrs.jsoc.Series.aia_lev1_euv_12s & attrs.jsoc.Notify(
                JSOC_EMAIL
            )

            logging.info("Performing Fido search with these parameters...")
            logging.info(f"Base attrs: {base_attrs}")
            logging.info(f"Inst attrs: {inst_attr}")
            fido_search_result = Fido.search(base_attrs, inst_attr)
            logging.info(f"Fido search result: {fido_search_result}")

            self.fido_search_result = fido_search_result
            self.fido_search_found_num = len(fido_search_result)

            if self.fido_search_found_num == 0:
                logging.warning("No results found in Fido search!")
            else:
                logging.info(f"Found {self.fido_search_found_num} results.")
        except Exception as e:
            logging.error(f"Error during Fido search: {e}")

    def validate_fits(self):
        """Validate FITS files to identify bad frames."""
        all_fits_paths = self.params.local_fits_paths()
        destroyed, missing, dark = self._validate_fits_files(all_fits_paths)
        self._print_validation_summary(destroyed, missing, dark, len(all_fits_paths))

    def _validate_fits_files(self, all_fits_paths):
        """Helper method to validate each FITS file."""
        destroyed, missing, dark = 0, 0, 0
        for local_fits_path in tqdm(
            all_fits_paths, desc="> Validating Fits Files", unit="imgs"
        ):
            if self._is_invalid_fits(local_fits_path):
                destroyed += 1
                missing += 1
        return destroyed, missing, dark

    def _is_invalid_fits(self, fits_path):
        """Check if a FITS file is invalid."""
        with fits.open(fits_path, ignore_missing_end=True) as hdul:
            img_type = hdul[1].header.get("IMG_TYPE", "").lower()
            if img_type == "dark" or not self._is_data_valid(hdul):
                return True
        return False

    def _is_data_valid(self, hdul):
        """Check if the data in the FITS file is valid."""
        frame = hdul[-1].data
        good_pix = np.sum(np.isfinite(frame))
        total_pix = frame.size
        return good_pix / total_pix >= 0.6

    def _print_validation_summary(self, destroyed, missing, dark, total_files):
        """Print a summary of the FITS file validation process."""
        if destroyed:
            logging.info(
                f"Validated: {total_files - destroyed}/{total_files}. Bad Frames: {destroyed}. Missing: {missing}. Dark: {dark}."
            )
        else:
            logging.info(f"Validated: {total_files}/{total_files}. No Bad Frames!")
