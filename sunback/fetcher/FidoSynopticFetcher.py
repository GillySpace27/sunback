import os
import sys
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from sunpy.net import Fido, attrs as a
import astropy.units as u
import warnings
from contextlib import redirect_stdout, redirect_stderr
from sunback.utils.time_util import parse_time_string_to_local
from sunback.fetcher.Fetcher import Fetcher
from sunback.fetcher.FidoFetcher import FidoFetcher
from sunback.fetcher.AIASynopticClient import AIASynopticData, AIASynopticClient
import asyncio
import aiohttp
import time
import random

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
    can_do_parallel = True  # Enable parallel processing

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.urls = []  # To store the list of URLs for parallel processing

    def fetch(self, params=None, quick=False, rp=None, verb=True):
        if verb is not None:
            self.verb = verb
        self.params.do_parallel_init = self.params.do_parallel and True
        self.params.do_parallel = False
        self.setup_fetcher(params, quick, rp)
        self.fido_get_fits(self.params.current_wave())
        self.params.do_parallel = self.params.do_parallel_init

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
        """Prepare the list of URLs for parallel downloading."""
        self.set_output_path()
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        # Extract URLs from the Fido search result
        self.urls = [result_item['url'] for result_item in self.fido_search_result[0]]

        # Use parallel_fits_series if parallel processing is enabled
        if self.params.do_parallel:
            self.parallel_fits_series()
        else:
            self.serial_fits_series()

    def parallel_fits_series(self):
        """Run parallel downloading using a process pool."""
        print("Running in Parallel Mode...", end="")
        self.init_pool_if_needed()

        try:
            # Using imap_unordered for parallel processing of each URL
            iter = self.params.multi_pool.imap_unordered(self.download_one_file, self.urls)

            pbar = self.init_pbar_now(total=len(self.urls))
            for _, result in enumerate(iter):
                pbar.update()
                if result is None:
                    self.skipped += 1
            print("Finished", flush=True)
        except Exception as e:
            print("Parallel Run Failed:", e)
            self.serial_fits_series()

    def download_one_file(self, url):
        """Download a single FITS file."""
        try:
            # Create the save path based on the output directory
            file_name = os.path.basename(url)
            save_path = os.path.join(self.out_path, file_name)

            # Perform the download
            asyncio.run(self.download_file_task(url, save_path))

            return True  # Indicate success
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None  # Indicate failure

    async def download_file_task(self, url, save_path):
        """Asynchronously download a single file."""
        async with aiohttp.ClientSession() as session:
            retries = 0
            while retries < 5:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            with open(save_path, 'wb') as f:
                                while chunk := await response.content.read(1024):
                                    f.write(chunk)
                            return
                        else:
                            print(f"Failed to download {url}. Status: {response.status}")
                            break
                except Exception as e:
                    print(f"Error downloading {url}: {e}")

                retries += 1
                await asyncio.sleep(0.5 * 2**retries)  # Exponential backoff

    def serial_fits_series(self):
        """Run the download process serially."""
        for url in tqdm(self.urls, desc="Downloading files"):
            self.download_one_file(url)

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