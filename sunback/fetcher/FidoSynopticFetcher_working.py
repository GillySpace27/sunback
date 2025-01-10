import os
import sys
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from sunpy.net import Fido, attrs as a
from parfive import Downloader
import astropy.units as u
import warnings
from contextlib import redirect_stdout, redirect_stderr
from sunback.utils.time_util import parse_time_string_to_local
from sunback.fetcher.Fetcher import Fetcher
from sunback.fetcher.FidoFetcher import FidoFetcher
from sunback.fetcher.AIASynopticClient import AIASynopticData, AIASynopticClient

# Constants
global_verbosity = False


def vprint(message, verbose=None, global_verbosity=global_verbosity, *args, **kwargs):
    if verbose or global_verbosity:
        print(message, *args, **kwargs)


class FidoSynopticFetcher(Fetcher):
    description = "Get FITS Files from the Internet using Fido"
    verbose = True
    filt_name = "Fido Synoptic Fetcher"
    batch_id = 0
    needed_files = None
    results = None
    temp_folder = None
    fits_path = None

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)

    def setup_fetcher(self, params=None, quick=False, rp=None):
        self.SubDownloader = None
        self.reprocess_mode(rp)
        self.params.load_preset_time_settings()
        self.set_output_path()  # Set up the output path using batching logic

    def fetch(self, params=None, quick=False, rp=None, verb=True):
        if verb is not None:
            self.verb = verb
        self.setup_fetcher(params, quick, rp)
        self.fido_get_fits(self.params.current_wave())

    def fido_get_fits(self, current_wave):
        """
        Fetches FITS files using Fido if they are not already cached.
        """
        print("Synoptic Fetcher")
        self.load(self.params, wave=current_wave)
        have_file = self.determine_image_path()  # Check if the file already exists

        time_integrator = type(self) not in (FidoFetcher, FidoSynopticFetcher)
        out_string = "\r v Fetching FITS Files: {}  ---------------------------------------------------  v"
        vprint(out_string.format(self.params.current_wave()), self.verb)

        need_file = self.params.download_files() and not have_file
        want_to_redo = self.reprocess_mode() and have_file

        if need_file or want_to_redo or time_integrator:
            self.print_load_banner(verb=self.verb)
            self.download_fits_series()
        else:
            prnt = "\b" if self.params.do_single else self.params.n_fits
            vprint(" *\n ^ Using {} Cached FITS Files".format(prnt), self.verb)

    def download_fits_series(self):
        self.params.define_range()
        self.fido_check_for_fits()
        if self.fido_search_found_num:
            self.fido_parse_result()
            self.download_files_with_retry()
            self.validate_download()
        else:
            print("\n     No Images Found\n")

    def fido_check_for_fits(self, verb=None):
        """
        Uses Fido to search for FITS files matching the specified parameters.
        """
        self.verb = self.verb or verb
        time_attr = a.Time(self.params.start_time, self.params.end_time)
        wave_attr = a.Wavelength(int(self.params.current_wave()) * u.angstrom)
        sample_attr = a.Sample(self.params.cadence_minutes())
        inst_attr = a.Instrument("AIA") & AIASynopticData()

        query = time_attr & wave_attr & sample_attr & inst_attr
        fido_search_result = Fido.search(query)

        if self.verb:
            print(fido_search_result)
        self.fido_search_result = fido_search_result
        self.fido_search_found_num = len(self.fido_search_result[0])

    def download_files_with_retry(self, max_retries=10):
        """Download the files using Fido with retry logic for missed files."""
        self.SubDownloader = Downloader(
            progress=True, max_conn=10, overwrite=self.reprocess_mode
        )
        self.set_output_path()  # Use the batching logic to determine the output path

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        pattern = f"{self.out_path}/{{file}}"
        print("THING: ", (pattern))
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.results = Fido.fetch(
                    self.fido_search_result,
                    # path=os.path.join(self.out_path, "{file}"),  # Corrected line
                    path=pattern,
                    downloader=self.SubDownloader,
                )
                if len(self.results) == self.fido_search_found_num:
                    break  # Stop retrying if all files are successfully downloaded
                else:
                    vprint(
                        f"Retrying download: attempt {retry_count + 1}/{max_retries}",
                        self.verb,
                    )
                    self.validate_and_cleanup_files()  # Validate the files and retry missing ones
                    retry_count += 1
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Max retries reached. Error: {e}")
                else:
                    print(f"Error encountered, retrying: {e}")

        self.multi_banner()

    def set_output_path(self):
        """Sets the output path for downloaded files using batch logic."""
        base_path = self.params.fits_directory()
        self.out_path = base_path

    def determine_image_path(self):
        """
        Checks if the expected FITS file exists in the output directory.
        Returns the file path if it exists, otherwise returns False.
        """
        current_wave = self.params.current_wave()
        file_list = os.listdir(self.out_path) if os.path.exists(self.out_path) else []

        for file_name in file_list:
            if current_wave in file_name:
                file_path = os.path.join(self.out_path, file_name)
                return file_path  # Return the full path if the file exists

        return False  # Return False if the file is not found

    def validate_and_cleanup_files(self):
        to_destroy = self.validate_fits()
        if to_destroy:
            self.destroy_files(to_destroy)

    def validate_fits(self):
        """Validate the downloaded FITS files and mark any corrupted files for redownload."""
        self.load_fits_paths()
        all_fits_paths = self.params.local_fits_paths()
        to_redownload = []

        for local_fits_path in tqdm(
            all_fits_paths, desc=" > Validating FITS Files", unit="imgs"
        ):
            delete = False
            with fits.open(local_fits_path, ignore_missing_end=True) as hdul:
                hdul.verify("silentfix+warn")
                img_type = hdul[1].header.get("IMG_TYPE", "").casefold()
                if img_type == "dark" or not np.isfinite(hdul[-1].data).any():
                    delete = True
            if delete:
                to_redownload.append(local_fits_path)

        # Attempt to redownload the corrupted or missing files at least once
        if to_redownload:
            vprint(f"Found {len(to_redownload)} files to redownload.", self.verb)
            self.redownload_missing_files(to_redownload)

        return to_redownload

    def redownload_missing_files(self, to_redownload):
        """Attempt to redownload the missing or corrupted files before marking them for destruction."""
        # Create a new Fido search result for the files to redownload
        redownload_result = Fido.search(
            a.Query([{"url": url} for url in to_redownload])
        )

        retry_results = Fido.fetch(
            redownload_result,
            path=os.path.join(self.out_path, ""),  # "{file}"),
            downloader=self.SubDownloader,
        )
        successfully_downloaded = [
            file for file in retry_results if os.path.exists(file)
        ]

        for file in successfully_downloaded:
            if file in to_redownload:
                to_redownload.remove(file)

        if to_redownload:
            vprint(
                f"Failed to redownload {len(to_redownload)} files. Marking them for deletion.",
                self.verb,
            )
            self.destroy_files(to_redownload)

    def destroy_files(self, to_destroy):
        for path in to_destroy:
            try:
                os.remove(path)
            except (PermissionError, FileNotFoundError):
                pass

    def get_start_and_end_times_from_result(self):
        all_times = []

        # Iterate over each QueryResponse in the UnifiedResponse
        for response in self.fido_search_result:
            if "Start Time" in response.colnames:
                times = response["Start Time"]
                all_times.extend(times)

        if all_times:
            all_times.sort()
            return all_times[0], all_times[-1]
        else:
            return None, None

    def fido_parse_result(self):
        self.start_time, self.end_time = self.get_start_and_end_times_from_result()
        try:
            begin_time = self.start_time.strftime("%I:%M:%S%p %m/%d/%Y").lower()
            end_time = self.end_time.strftime("%I:%M:%S%p %m/%d/%Y").lower()
        except (ValueError, AttributeError):
            begin_time = str(self.start_time)
            end_time = str(self.end_time)

        self.extra_string = "from {} to {}".format(begin_time, end_time)

        # Estimate total download size
        total_size_bytes = 0
        estimated_size_per_file = 0.5 * 1024 * 1024  # 0.5 MB in bytes
        total_size_bytes = self.fido_search_found_num * estimated_size_per_file
        total_size_mb = total_size_bytes / (1024 * 1024)  # Convert bytes to megabytes

        print(f"Using output path: {self.out_path}")

        if self.fido_search_found_num > 200:
            response = input(
                "\nThis download will be approximately {:.2f} MB. Do you still want to download all {} images? [y]/n > ".format(
                    total_size_mb, self.fido_search_found_num
                )
            )
            if "n" in response.casefold():
                print("Stopping!\n\n")
                raise StopIteration

    def multi_banner(self):
        print("\r   [\\~~~~~~~~~~~~~~~~~~~~~~~~~~~FIDO~~~~~~~~~~~~~~~~~~~~~~~~~~~//]\n")
        if self.results and len(self.results) == self.fido_search_found_num:
            print(
                f"\r ^     Successfully Downloaded all {len(self.results)} Files\n",
                flush=True,
            )
        elif self.results:
            print(
                f" ^     Downloaded {len(self.results)} Files out of {self.fido_search_found_num}\n",
                flush=True,
            )
        else:
            print(" ^     Unable to Download...Try again Later.")
            raise ConnectionRefusedError(" Unable to Download...Try again Later.")

    def validate_download(self):
        # Implement any validation after download if needed
        pass
