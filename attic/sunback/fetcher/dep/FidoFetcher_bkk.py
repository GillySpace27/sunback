import os
from time import strptime, mktime
import sys
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from drms import DrmsExportError
from sunpy.net import Fido, attrs
from parfive import Downloader
import astropy.units as u
import datetime
from sunback.utils.time_util import (
    parse_time_string_to_local,
    define_time_range,
    define_recent_range,
)
from sunback.fetcher.Fetcher import Fetcher
from sunback.processor.SunPyProcessor import AIA_PREP_Processor
from sunpy.coordinates.sun import carrington_rotation_time

# Constants
default_base_url = "http://jsoc1.stanford.edu/data/aia/synoptic/mostrecent/"
jsoc_email = "chris.gilly@colorado.edu"
global_verbosity = False


def vprint(message, verbose=None, global_verbosity=global_verbosity, *args, **kwargs):
    if verbose or global_verbosity:
        print(message, *args, **kwargs)


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
        self.SubDownloader = None
        self.reprocess_mode(rp)
        self.params.load_preset_time_settings()

    ## Main Fetch Logic
    def fetch(self, params=None, quick=False, rp=None, verb=True):
        if verb is not None:
            self.verb = verb
        """ Find the Most Recent Images """
        self.__init__(params, quick, rp)
        # self.verb = True
        self.fido_get_fits(self.params.current_wave(), temp=self.params.do_temp)

    def cleanup(self):
        # self.fido_download_fits_ensured(hold=False, temp=True)
        # self.delete_temp_folder_items(delete_folder_too=True)
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
        pass

    def enumerate(self):
        # for fits_path in self.params.local_fits_paths():
        #     print(fits_path)
        pass

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
        # Get start and end times for the Carrington rotations
        start_time = carrington_rotation_time(carrington_start).to_datetime()
        end_time = carrington_rotation_time(carrington_end).to_datetime()

        # Calculate time step in minutes per frame
        time_delta = (end_time - start_time) / num_frames  # * 24 * 60

        # Generate the list of times for each frame
        times = [start_time + i * time_delta for i in range(num_frames)]
        return times

    def fido_get_fits(self, current_wave, temp=False, num_frames=2):
        self.load(self.params, wave=current_wave)
        have_file = self.determine_image_path() is not False
        out_string = "\r v Fetching Fits Files: {}  ---------------------------------------------------  v"
        vprint(out_string.format(self.params.current_wave()), self.verb)
        need_file = self.params.download_files() and not have_file
        want_to_redo = self.reprocess_mode() and have_file

        if need_file or want_to_redo:
            self.print_load_banner(verb=self.verb)

            if self.params.carrington():
                self.params.carrington_start = self.params.carrington()[0]
                self.params.carrington_end = self.params.carrington()[1]
                self.params.num_frames = self.params.carrington()[-1]
                # Use Carrington rotations to define the time range
                times = self.carrington_to_time(
                    self.params.carrington_start,
                    self.params.carrington_end,
                    self.params.num_frames,
                )
                # Define the time range from the first to last time
                self.params.start_time = times[0]
                self.params.end_time = times[-1]
                self.params.time_steps = times  # Pass the list of times

                # Define time range from the first to last time
                self.params.unpack_time_strings(times[0], times[-1])
                self.params.time_steps = times  # Pass the list of times

            self.params.define_range()
            self.fido_check_for_fits()
            self.fido_download_fits_ensured(temp=temp)
        else:
            prnt = self.params.n_fits if not self.params.do_single else "\b"
            vprint(" *\n ^ Using {} Cached Fits Files".format(prnt), self.verb)

    def fido_check_for_fits(self, verb=None):
        """Find the science images."""
        from astropy import units as u

        self.verb = self.verb or verb
        vprint(
            "\n *   Looking for Images of {} from {} to {} with {:0.3} or {:0.3} cadence...".format(
                self.params.current_wave(),
                self.params.start_time_string,
                self.params.end_time_string,
                self.params.cadence_minutes().to(u.d),
                self.params.cadence_minutes().to(u.s),
            ),
            flush=True,
            end="",
            verb=self.verb,
        )

    def fido_parse_result(self):
        """Examine the search results"""
        self.start_time, self.end_time = self.get_start_and_end_times_from_result()

        try:
            begin_time = parse_time_string_to_local(self.start_time, 4)[0]
            end_time = parse_time_string_to_local(self.end_time, 4)[0]
        except ValueError:
            # Handle the error or log it
            pass
        self.extra_string = "from {} to {}".format(begin_time, end_time)

        if self.fido_search_found_num > 1:
            vprint(
                "\n *      Search Found {: 3} Images {}...".format(
                    self.fido_search_found_num, self.extra_string
                ),
                flush=True,
                verb=self.verb,
            )
        elif self.fido_search_found_num == 1:
            vprint(
                "\n *      Search Found  {: 3} Image  at  {}...".format(1, begin_time),
                flush=True,
                verb=self.verb,
            )
            vprint(
                " *                             End = {}...".format(
                    self.params.end_time_string
                ),
                flush=True,
                verb=self.verb,
            )
        else:
            vprint("\n *      Search Found Nothing")
            raise FileNotFoundError

        while len(self.name) < 4:
            self.name = "0" + self.name

        if self.fido_search_found_num > 200 and False:
            response = input(
                "Do you still want to download all {} images? [y]/n > ".format(
                    self.fido_search_found_num
                )
            )
            if "n" in response.casefold():
                print("Stopping!")
                raise StopIteration
            print("Continuing. ", end="")

    def store_requests(self):
        try:
            response = self.fido_search_result.get_response(0)
        except AttributeError:
            response = self.fido_search_result
        self.needed_files = response
        self.num_files_needed = self.needed_files._numfile

    def fido_download_fits_ensured(self, temp=False, hold=False, ensured=True):
        """Download the files from fido_search_result"""

        self.SubDownloader = Downloader(progress=True, max_conn=10, overwrite=False)

        self.out_path = (
            self.params.temp_directory() if temp else self.params.fits_directory()
        )
        print("Out Path: ", self.out_path)
        self.store_requests()

        main_stdout = sys.stdout

        if not hold:
            loc = os.path.join(self.params.temp_directory(), "log.txt")
            # with open(loc, mode="w+") as sys.stdout:
            self.verb = False
            print(" **       Fido Fetching...")
            print(
                "\r \n   [/~~~~~~~~~~~~~~~~~~~~~~~~~~~FIDO~~~~~~~~~~~~~~~~~~~~~~~~~~~\\]"
            )
            try:
                self.results = Fido.fetch(
                    self.needed_files, path=self.out_path, downloader=self.SubDownloader
                )
            except DrmsExportError as e:
                print(e)
                self.results = []

            self.n_fits = len(self.results)
            if ensured:
                self.results = self.fido_multi_download()
            self.multi_banner()
            # self.results = copy.copy(results)

            # self.params.params_path()
            # self.params.save_to_txt()

            sys.stdout = main_stdout

            return self.results

    def fido_multi_download(self):
        self.n_fits = -1
        # self.validate_fits()

        ii = 0
        while self.n_fits != self.fido_search_found_num and ii < 10:
            self.results = Fido.fetch(
                self.results, path=self.out_path, downloader=self.SubDownloader
            )
            self.n_fits = len(self.results)
            to_destroy = False  # self.validate_fits()
            if to_destroy:
                self.destroy_files(to_destroy)
                self.n_fits = -1

            ii += 1

        self.n_fits = len(self.results)
        if self.params.do_single:
            self.n_fits = 1
        return self.results

    def destroy_files(self, to_destroy=[]):
        if to_destroy:
            for path in to_destroy:
                break
                self.remove_files(path)
        self.n_fits = len(self.load_fits_paths())

    @staticmethod
    def remove_files(local_fits_path):
        # if local_fits_path in self.params.local_fits_paths():
        #     self.params.local_fits_paths().remove(local_fits_path)

        dir = os.path.dirname(local_fits_path)
        directory = dir.replace("fits", "png\\mod")
        file = os.path.basename(local_fits_path)
        png_file = file.replace(".fits", ".png")
        png_path = os.path.join(directory, png_file)

        dead_paths = [
            local_fits_path,
            png_path,
            png_path.replace("mod", "cat"),
            png_path.replace("mod", "orig"),
        ]
        deleted_files = 0
        print()
        for path in dead_paths:
            try:
                print("    Deleting a File...", end="")
                os.remove(path)
                if os.path.exists(path):
                    raise FileExistsError(path)
                print("Success!")
                deleted_files += 1
            except PermissionError as e:
                print(" Couldn't Access/Delete\n {}".format(path))
                print(e)
                # raise e
            except FileExistsError as e:
                print(
                    " File was still there after attempted deletion\n{}".format(
                        os.path.basename(path)
                    )
                )
                # print(e)
            except FileNotFoundError as e:
                # print(" Not Found to Delete\n {}".format(path))
                pass
            except Exception as e:
                print(" Failed to Delete\n {}".format(path))
                print(e)
                1 + 1
            print("Actually Deleted {} Files".format(deleted_files))

    # Time Related Things #########################################

    def get_start_and_end_times_from_result(self):
        # self.verb = True
        try:
            all_times = self.fido_search_result.get_response(0)
        except AttributeError as e:
            all_times = self.fido_search_result

        start_time_list = []
        # end_time_list = []
        if len(all_times) == 1:
            all_times = all_times[0]

        for result in all_times:
            try:
                try:
                    start_time_list.append(result["T_REC"])
                except Exception as e:
                    start_time_list.append(result["T_REC"][0])
                    raise e
            except KeyError:
                start_time_list.append(result["Start Time"].value)

            # end_time_list.append(result.time.pointing_end)

        times = sorted(start_time_list)
        time_start = times[0]
        time_end = times[-1]
        # ii=0
        # while time_start[-3:-1] < self.start_time[-2:]:
        #     time_start = times[ii]
        #     ii+=1
        # for t in range(ii-1):
        #     self.fido_search_result[0].remove_row(0)
        self.fido_search_found_num = self.fido_search_result.file_num

        return time_start, time_end

    # Printing #####################################################

    def multi_banner(self):
        print("\r   [\\~~~~~~~~~~~~~~~~~~~~~~~~~~~FIDO~~~~~~~~~~~~~~~~~~~~~~~~~~~//]\n")
        if self.n_fits == self.fido_search_found_num:
            print(
                "\r ^     Successfully Downloaded all {} Files\n".format(self.n_fits),
                flush=True,
            )
        elif self.n_fits:
            print(
                " ^     Downloaded {} Files out of {}\n".format(
                    self.n_fits, self.fido_search_found_num
                ),
                flush=True,
            )
        else:
            print(" ^     Unable to Download...Try again Later.")
            raise (ConnectionRefusedError(" Unable to Download...Try again Later."))

        self.super_flush()

    # Validation
    def validate_download(self):
        # Currently not running, probably for the best
        if self.params.do_prep:
            print("AIA Prepping...")
            self.params.speak_save = False
            AIA_PREP_Processor(params=self.params, rp=True).process()

    def validate_fits(self):
        import numpy as np

        # if True:
        #     return []
        self.load_fits_paths()
        all_fits_paths = self.params.local_fits_paths()
        n_fits = len(all_fits_paths)
        destroyed = 0
        dark = 0
        missing = 0
        to_destroy = []
        to_redownload = []
        # print('Validation is Running')
        for local_fits_path in tqdm(
            all_fits_paths, desc=" > Validating Fits Files", unit="imgs"
        ):
            # for local_fits_path in all_fits_paths:
            delete = False
            if local_fits_path:
                with fits.open(local_fits_path, ignore_missing_end=True) as hdul:
                    hdul.verify("silentfix+warn")
                    # self.rename_initial_frames(hdul) # This might not work
                    # TEST 1 - IS IT A DARK FRAME?
                    img_type = hdul[1].header["IMG_TYPE"]
                    if img_type.casefold() == "dark":
                        delete = True
                        dark += 1
                        # print("Dark Image Detected")
                    if not delete:
                        mean_value = hdul[1].header["DATAMEDN"]
                        quality_value = hdul[1].header["QUALITY"]
                        if mean_value < 50 or quality_value > 100:
                            delete = True
                            dark += 1
                    if not delete:
                        # TEST 2 - IS FILLED WITH NULLS?
                        frame = hdul[-1].data
                        good_pix = np.sum(np.isfinite(frame))
                        total_pix = frame.shape[0] * frame.shape[1]
                        good_percent = good_pix / total_pix
                        # Dark Frame had 0.37
                        if good_percent < 0.6:
                            # print("Good Percent: {:0.4f}".format(good_percent))
                            delete = True
                            missing += 1
                            # print("Missing Data Detected")

                if delete:
                    to_destroy.append(local_fits_path)
                    to_redownload.append(local_fits_path)
                    destroyed += 1
                    n_fits -= 1

        # self.destroy_files(to_destroy)
        # self.fido_get_fits()
        if destroyed:
            print(
                "\r      >>Fits Files Validated: {}, Bad Frames: {}\n VV  {} Dark, {} Missing<<\n".format(
                    n_fits, destroyed, missing, dark
                )
            )
        else:
            print(
                "\r      >>Fits Files Validated: {0}/{0}. No Bad Frames!<<".format(
                    n_fits
                )
            )
        #
        # if missing:
        #     print(" ii          > {} missing data".format(missing))
        # if dark:
        #     print(" ii          > {} dark frames".format(dark))
        #
        return to_destroy

        # try:
        #     self.set_output_paths()
        # except:
        #     set_output_paths(self)
        # self.list_requested_files()
        # self.local_fits_paths = list_files_in_directory(self.fits_folder)

        # pass

        # if self.params.delete_old():
        #     self.remove_all_old_fits_pngs()
        #     self.remove_all_old_pngs()
        #
        # working = False
        # if working:
        #     self.validate_fits()
        #     self.redownload_bad_fits()
        #
        # self.fido_download_fits_ensured()

        # self.find_missing_images()
        # self.get_missing_images()

    def list_requested_files(self):
        self.requested_files = []
        self.requested_response = []
        for ii in np.arange(self.fido_search_found_num):
            self.requested_files.append(
                self.fido_search_result.get_response(0)[ii]["fileid"].casefold()
            )
            self.requested_files.append(
                self.fido_search_result.get_response(0)[ii]["time"]["start_timestamp"]
            )

    @staticmethod
    def parse_filename_to_time(local_file):
        try:
            ifirst = 13 if "94" in local_file else 14
            stub = local_file[ifirst:-20]
            fmt_A = "%Y_%m_%dt%H_%M_%S"
            fmt_B = "%Y%m%d%H%M%S"
            # return stub.replace(['_', 't'], '')
            return datetime.datetime.strptime(stub, fmt_A).strftime(fmt_B)
        except:
            stub = local_file[3:-10].replace("_", "")
            return stub

    @staticmethod
    def out_of_range(hdul):
        print("A")
        pass
        return False
