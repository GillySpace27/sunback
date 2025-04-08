import os
from sunpy.net import Fido, attrs as a
from parfive import Downloader
import astropy.units as u
from sunback.fetcher.Fetcher import Fetcher
from sunback.fetcher.AIASynopticClient import AIASynopticData
import shutil
class FidoSynopticFetcher(Fetcher):
    description = "Get FITS Files from the Internet using Fido"

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.out_path = self.params.fits_directory()
        os.makedirs(self.out_path, exist_ok=True)
        self.results = None

    def fetch(self, params=None, quick=False, rp=None, verb=True):
        self.setup_fetcher(params, quick, rp)
        self.fido_get_fits(self.params.current_wave())

    def setup_fetcher(self, params=None, quick=False, rp=None):
        self.reprocess_mode(rp)
        self.params.load_preset_time_settings()

    def fido_get_fits(self, current_wave):
        # print("Synoptic Fetcher")
        self.load(self.params, wave=current_wave)
        if self.params.download_files():
            if self.reprocess_mode():
                shutil.rmtree(self.out_path)
            os.makedirs(self.out_path, exist_ok=True)
            self.download_fits_series()
        else:
            print("Using cached FITS files")

    def download_fits_series(self):
        self.params.define_range()
        search_result = self.fido_search()
        if search_result:
            self.download_files(search_result)
        else:
            print("No Images Found")

    def fido_search(self):
        time_attr = a.Time(self.params.start_time, self.params.end_time)
        wave_attr = a.Wavelength(int(self.params.current_wave()) * u.angstrom)
        sample_attr = a.Sample(self.params.cadence_minutes())
        inst_attr = a.Instrument("AIA") & AIASynopticData()

        query = time_attr & wave_attr & sample_attr & inst_attr
        search_result = Fido.search(query)
        if len(search_result[0]) == 0:
            return None
        return search_result

    def download_files(self, search_result, max_retries=5):
        downloader = Downloader(max_conn=5, progress=True, overwrite=False)
        pattern = os.path.join(self.out_path) #, "{file}")

        for retry in range(max_retries):
            # Enqueue files that are missing
            for record in search_result[0]:

                filename = record['fileid'].split('/')[-1]
                filepath = os.path.join(self.out_path, filename)
                if not os.path.exists(filepath):
                    downloader.enqueue_file(record['url'], path=pattern)

            # Start download
            results = downloader.download()

            # Check for failed downloads
            if results.errors:
                print(f"Retry {retry + 1}/{max_retries} failed for {len(results.errors)} files.")
                search_result = Fido.search(*results.errors)
            else:
                print(f"All {len(results.data)} files downloaded successfully.")
                break
        else:
            print("Some files could not be downloaded after maximum retries.")