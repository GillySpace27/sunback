import os
import sys
from copy import copy, deepcopy
from datetime import datetime
from os import listdir, getcwd, makedirs
from os.path import join, dirname, abspath, isdir, basename
from pickle import PicklingError
from random import choices

import sunpy

print(sunpy.__version__)

from scipy.stats import stats
# from sunpy.errors import SunPyError

# from sunpy.map.header_helper import MetaDataMissingError, MetaDataParseError
import sunpy.map

# from sunpy.map.errors import MetaDataMissingError, MetaDataParseError
import astropy.units as u
from time import sleep, strptime, mktime
import time

import cv2
import numpy as np
import sunpy
from sunpy.map import Map as mp
from astropy.io.fits.verify import VerifyError

from sunback.science.color_tables import aia_color_table

# import cv2
from astropy.io import fits
from tqdm import tqdm

import matplotlib

# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

verb = True


def vprint(in_string, *args, **kwargs):
    if verb:
        print(in_string, *args, **kwargs)


class Processor:
    """Top Level Class"""

    # name = 'data'
    in_name = None
    filt_name = "Base Processor Class"
    out_name = batch_name = name = "default_name"
    description = "Use an Unnamed Processor"
    run_type = "General Base Processor Class"
    progress_stem = " *    {} {}"
    progress_verb = "Processing"
    progress_unit = "files"
    progress_string = progress_stem.format(progress_verb, progress_unit)
    finished_verb = "Processed"
    run_type_string = "Default Actions"
    out_path = None
    print_once = False
    style_mode = "all"
    do_png = False
    quietly = True
    params = None
    # current_wave = 'rainbow'
    proc_name = None
    modified_image = None
    raw_image = None
    n_fits = None
    n_imgs = None
    ii = 0
    cmap = "plasma"
    # all_wavelengths = ['0211', '0304', '0131', '0335']

    # waves_to_do = all_wavelengths
    dont_ignore = False
    keyframes = []
    _reprocess_mode = None
    base_fits_dir = None
    base_imgs_dir = None
    base_absolute = None
    save_to_fits = True
    can_use_keyframes = False
    can_do_parallel = False
    this_file_name = os.path.basename(__file__)
    paper_out = []

    curves_have_been_loaded = False
    limb_radius_from_header = None
    fits_folder = None
    abs_min_scalar = None
    curve_out_array = None
    ensured = False
    hdu_name_list = None
    file_basename = None
    image_data = None
    changed_flat = None
    can_use_keyframes = False
    use_keyframes = None
    skipped = 0
    all_file_paths = None
    n_all_frames = None
    n_do_frames = None
    long_list = []
    fits_path = None
    first_hIndex = 0
    short_list = []
    out_dtype = np.float32
    frame_name = None
    init_active = False
    do_print_success = True

    def __init__(self, params=None, quick=False, rp=None, in_name=None):
        self.binInds_forpoints = None
        self.ratio_factor_for_radius = None
        self.limb_radius_from_file_shrunken = None
        self.vig_radius_rr = 1000
        self.binRR = None
        self.binII = None
        self.binYY = None
        self.binXX = None
        self.shrink_F = 1
        self.limb_radius_from_fit_shrunken = None
        self.limb_radius_from_header_shrunken = None
        self.limb_radius_already_found = False
        self.lCut = None
        self.hCut = None
        self.output_abscissa = None
        self.wave = None
        self.binInds = None
        self.bin_rez = None
        self.radBins = None
        self.radBins_xy = None
        self.radBins_ind = None
        self.binMax = None
        self.binMin = None
        self.binAbsMax = None
        self.binAbsMin = None
        self.binAbsMin = None
        self.binBox = []

        self.limb_radius_from_fit_shrunken = None
        self.radius = None
        self.do_split = False
        self.loud_tic = False
        self.duration = None
        self.tm = time.time()
        self.raw_map = None
        self.img_path = None
        self.vignette_mask = None
        self.in_name = in_name or self.in_name
        self.header = None
        self.reprocess_mode(rp)
        self.load(params, quick=quick)
        # Initialize a list to collect failed FITS files
        self.failed_fits = []
        # Optionally, initialize a counter for fixed FITS files
        self.fixed_fits_count = 0
        # self.print_once = True
        # self.tic()
        if self.params:
            self.run_type_str = "\\item  {}".format(self.this_file_name, self.run_type)
            self.paper_out.append(self.run_type_str)
        else:
            raise ModuleNotFoundError

    @staticmethod
    def plan(self, durList=None, end=False):
        """Find the name of this processor and print"""
        if end:
            # print(self)
            print(
                "      {:15} : \t{:20} : \t{:15}".format(
                    self.filt_name, self.description, np.round(self.duration, 4)
                )
            )
        elif self.filt_name is not None:
            print("      {:15} : \t{:20}".format(self.filt_name, self.description))

    #  def plan(self, durList=None, end=False):
    #     """Find the name of this processor and print"""
    #     try:
    #         if self.duration in [None, -1.0]:
    #             if durList is not None and len(durList) > 0:
    #                 pop = float(durList.pop(0))
    #                 self.duration = str(round(pop, 4))
    #             else:
    #                 self.duration = -1.0
    #     except AttributeError:
    #         self.duration = -1.0

    #     if self.filt_name is not None:
    #         print('      {:15} : \t{:20} : \t{:15}'.format(self.filt_name, self.description, self.duration))
    def put(self, params=None):
        self.process(params)
        pass

    def fetch(self, params=None):
        self.process(params)

    def reprocess_mode(self, flag=None):
        if flag is not None:
            if type(flag) is not bool:
                if "skip" == flag:
                    flag = False
                if "redo" == flag:
                    flag = True
            self._reprocess_mode = flag

        return self._reprocess_mode

    # def __getstate__(self):
    #     self_dict = self.__dict__.copy()
    #     try:
    #         del self_dict['multi_pool']
    #     except KeyError:
    #         pass
    #     return self_dict
    ##############################################################
    ## M1: Look for files in a directory and return their paths ##
    ##############################################################
    def find_frames_at_path(self, fits_path):
        """Determine which frames exist in a given fits file"""
        with fits.open(fits_path, cache=False, reprocess_mode="update") as hdul:
            self.hdu_name_list = self.list_hdus(hdul)
        self.good_frames = [x for x in self.hdu_name_list if self.image_is_plottable(x)]
        return self.good_frames

    def load(
        self,
        params=None,
        fits_directory=None,
        imgs_directory=None,
        absolute=True,
        in_name=None,
        out_name=None,
        batch_name=None,
        quietly=True,
        wave=None,
        quick=False,
    ):
        """M1
        Create and return two lists
            the fits files in params.fits_directory()
            the img  files in params.imgs_top_directory()
        """
        verb = not quietly
        self.params = params or self.params

        self.set_names(in_name, out_name, batch_name, quietly)
        self.progress_string = self.progress_stem.format(
            self.progress_verb, self.progress_unit
        )
        fits_paths, imgs_paths = None, None
        if self.params is not None:
            wave = wave or self.params.current_wave()
            #  Refresh Params and Load Paths
            self.name = self.params.batch_name(batch_name)
            self.super_flush()
            if wave:
                self.params.set_current_wave(wave)
            self.select_keyframe_subset()
            # self.params.create_subdirectories()  #Gender
            fits_paths, imgs_paths = self.load_paths(verb)

        self.set_base_directories(fits_directory, imgs_directory, absolute)
        self.super_flush()
        return fits_paths, imgs_paths

    # def clean_directory(self):
    #     to_rep = "D:/"
    #     if not self.params.base_directory()[0] == to_rep[0]:
    #         self.params.base_directory(self.params.base_directory().replace(to_rep,""))
    #         if self.out_path:
    #             self.out_path = self.out_path.replace(to_rep,"")

    # Define Targets
    def set_names(self, in_name=None, out_name=None, name=None, quietly=None):
        """Store the batch names into self"""
        if in_name:
            self.in_name = in_name
        if out_name:
            self.out_name = out_name
        if name:
            self.name = name
        if quietly:
            self.quietly = quietly

    def set_base_directories(
        self, fits_directory=None, imgs_directory=None, absolute=None
    ):
        """Store the directories into self"""
        if fits_directory:
            self.base_fits_dir = fits_directory
        elif self.base_fits_dir is None:
            self.base_fits_dir = self.params.fits_directory()

        if imgs_directory:
            self.base_imgs_dir = imgs_directory
        elif self.base_imgs_dir is None:
            self.base_imgs_dir = self.params.mods_directory()

        if absolute is not None:
            self.base_absolute = absolute

    def make_temp_dir(self, img_path):
        if self.params.do_single and img_path:
            temp_top_dir = os.path.dirname(self.params.temp_directory())
            this_name = os.path.basename(img_path)[:-5]
            self.params.temp_directory(os.path.join(temp_top_dir, this_name, "RHT"))
        os.makedirs(self.params.temp_directory(), exist_ok=True)
        self.params.do_temp = True
        return self.params.temp_directory()

    def load_paths(self, verb=False):
        """Determines and lists the files that exist in the given directories"""
        fits_paths, imgs_paths = self.load_fits_paths(), self.load_imgs_paths()
        self.print_load_banner(verb)
        return fits_paths, imgs_paths

    def print_load_banner(self, verb=False):
        if self.n_fits + self.n_imgs > 0 and verb:
            print(
                "\r v {}...  ------------------------------------------------  v".format(
                    self.filt_name
                ),
                flush=True,
            )
            sys.stdout.flush()
            if self.finished_verb.casefold() in ["summed"]:
                exp = self.params.exposure_time_seconds()
                print(
                    " *    Exposure Time is {} seconds, which is {:0.2f} frames".format(
                        exp, exp / 12
                    )
                )
            print(
                "\r +    {}: {}, Redo = {}".format(
                    self.progress_verb,
                    self.params.current_wave(),
                    self.reprocess_mode(),
                )
            )
            # vprint("\r +    Using {} fits and {} imgs from {}\n".format(self.n_fits, self.n_imgs, self.params.base_directory()))

    def load_fits_paths(self, absolute=True, ext=".fits"):
        """Creates a List of the existant fits files in the fits_directory"""
        self.fits_folder = (
            self.params.temp_directory()
            if self.params.do_temp
            else self.params.fits_directory()
        )
        paths, abs_paths = self.__find_ext_files_in_directory(self.fits_folder, ext)
        out_paths = self.params.local_fits_paths(abs_paths if absolute else paths)
        self.n_fits = self.params.n_fits = len(self.params.local_fits_paths())
        if not self.quietly:
            (
                "   Found {} {} Files in {}".format(
                    self.params.n_fits, ext, self.params.fits_directory()
                )
            )
        return out_paths

    def load_imgs_paths(self, absolute=True, ext=".png"):
        """Creates a List of the existant img files in the imgs_top_directory"""
        paths, abs_paths = self.__find_ext_files_in_directory(
            self.params.mods_directory(), ext
        )
        out_paths = self.params.local_imgs_paths(abs_paths if absolute else paths)
        self.n_imgs = self.params.n_imgs = len(self.params.local_imgs_paths())
        if not self.quietly:
            print(
                "   Found {} {} Files in {}".format(
                    self.params.n_imgs, ext, self.params.imgs_top_directory()
                )
            )
        return out_paths

    @staticmethod
    def __find_ext_files_in_directory(directory, ext=".fits"):
        """Returns the paths to matching ext files in given directory"""
        if not os.path.exists(directory):
            return [], []
            # makedirs(directory, exist_ok=True)
        ext_paths = [path for path in listdir(directory) if ext in path]
        abs_ext_paths = [join(directory, path) for path in ext_paths]
        return ext_paths, abs_ext_paths

    def load_fits_image(self, fits_path=None, in_name=None):
        """open the fits file and grab_obj the necessary data"""

        if fits_path is not None:
            self.fits_path = os.path.normpath(fits_path)
        if self.fits_path is None:
            self.fits_path = self.params.local_fits_paths()[0]

        if self.params.fits_path is None:
            self.params.fits_path = self.fits_path

        if type(in_name) in [str]:
            in_name = in_name.casefold()
        frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(
            self.fits_path, in_name
        )
        if frame is not None and self.header.get("IMG_TYPE", "").casefold() != "dark":
            self.params.raw_name = self.frame_name
            self.params.raw_image = np.asarray(frame, dtype=np.float32) + 0.0
            self.params.raw_image2 = np.asarray(frame, dtype=np.float32) + 0.0

            if (
                self.params.modified_image is None
                or self.params.modified_image.size == 1
                or self.params.do_single == False
            ):
                self.params.modified_image = copy(self.params.raw_image) + 0

            self.params.current_wave(wave)
            self.params.cmap = "greys"  # or aia_color_table(int(wave) * u.angstrom)
            self.image_data = str(wave), self.fits_path, t_rec, frame.shape
            self.file_basename = basename(self.fits_path)
            if center is not None:
                self.set_centerpoint(center)
            self.params.image_data = self.image_data
            return True
        else:
            # print("Skipped Fits!")
            pass
            # if img_type.casefold() == 'dark':
            #     self.delete_fits_and_png(fits_path)
            return False

    def delete_fits_and_png(self, fits_path, dopng=True):
        # fitsPath = join(self.fits_folder, filename[:-5] + '.fits')
        pngPath = fits_path.replace("fits", "png")
        try:
            os.remove(fits_path)
        except PermissionError as e:
            print(e)
        if dopng:
            try:
                os.remove(pngPath)
            except FileNotFoundError as e:
                print(e)
                pass

    def plot_two(self, name="Algorithm Result", bounds=None):
        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True, num=name)

        org = self.params.raw_image
        mod = self.params.modified_image

        # self.view_raw(fig, ax0)
        ax0.imshow(org, cmap=self.params.cmap)
        ax1.imshow(mod, cmap=self.params.cmap)

        if bounds is None:
            ax1.set_xlim((0, 1500))
            ax1.set_ylim((600, 2000))

        # else:
        #     ax1.set_xlim((3400,4000))
        #     ax1.set_ylim((2300,3200))

        ax0.set_title("lev1p0")
        ax1.set_title("Changed")

        plt.tight_layout()

        plt.show()

    def peek(self, img):
        fig, ax = plt.subplots()
        ax.set_title("Peek Frame")
        print("\rThe total summed value of the array is {}".format(np.nansum(img)))
        print(
            "{} percent of the entries are finite".format(
                np.nansum(np.isfinite(img)) / (self.params.rez**2)
            )
        )
        ax.imshow(img, interpolation=None, origin="lower", cmap=self.params.cmap)
        plt.show(block=True)

    def prep_one(self, img):
        minmin = np.nanmin(img)
        return img - minmin

    def set_centerpoint(self, center):
        """Parse the centerpoint and ensure correct scaling"""
        self.params.center = center
        image_edge = self.params.raw_image.shape
        center_given = np.asarray((np.abs(self.params.center)), dtype=float)

        Top_Tolerance = 0.65
        Bottom_Tolerance = 0.35
        count = 0
        # while count < 10:
        #     ratio = center_given / np.sum(image_edge, axis=-1)
        #     if np.array(ratio > Top_Tolerance).any():
        #         center_given *= 0.5
        #     elif np.array(ratio < Bottom_Tolerance).any():
        #         center_given *= 2
        #     else:
        #         break
        #     count += 1
        self.params.center = center_given

    def select_keyframe_subset(self):
        """Sets the list of which frames get used as keyframes
        This function only runs once, sort of an __init__
        """
        # if self.dont_ignore:
        self.use_keyframes = (
            self.params.fixed_cadence_keyframes()
            or self.params.fixed_number_keyframes()
        ) and self.can_use_keyframes
        if self.use_keyframes:
            self.keyframes = self.pick_keyframes()
        else:
            self.keyframes = self.pick_keyframes(use_all=True)
        pass
        # self.dont_ignore = False

    def pick_keyframes(self, use_all=False):
        """Decide which frames to use in the analysis"""
        # self.load(self.params, wave=self.params.current_wave)
        self.params.set_current_wave()
        if self.all_file_paths in [None, []]:
            self.all_file_paths = self.load_fits_paths()
        self.long_list = copy(self.all_file_paths)
        self.n_all_frames = len(self.long_list)
        n_paths = len(self.long_list)
        if self.n_all_frames < 100:
            use_all = True

        if use_all:
            self.short_list = self.long_list

        elif self.params.fixed_cadence_keyframes():
            # Fixed Cadence of one out of every {} frames
            self.short_list = self.long_list[:: self.params.fixed_cadence_keyframes()]

        elif self.params.fixed_number_keyframes():
            #  Fixed Number of Keyframes
            skip = max(n_paths // self.params.fixed_number_keyframes(), 1)
            self.short_list = self.long_list[::skip]
        self.n_do_frames = len(self.short_list)

        # percent_too_low = self.n_do_frames / self.n_all_frames < 0.25
        # number_of_frames_too_low = self.n_do_frames < 5
        # if number_of_frames_too_low:
        #     if percent_too_low:
        #         pass
        #     else:
        #         pass
        #

        return self.short_list

    def init_statistics(self):
        """Initialize the statistical arrays"""
        # dprint("init_statistics")

        self.bin_rez = np.max(self.binInds) + 10
        self.radBins = [[] for x in np.arange(self.bin_rez)]
        self.radBins_xy = [[] for x in np.arange(self.bin_rez)]
        self.radBins_ind = [[] for x in np.arange(self.bin_rez)]

        self.binMax = np.empty(self.bin_rez)
        self.binMin = np.empty(self.bin_rez)
        self.binAbsMax = np.empty(self.bin_rez)
        self.binAbsMin = np.empty(self.bin_rez)
        self.binAbss = np.arange(self.bin_rez)

        self.binMax.fill(np.nan)
        self.binMin.fill(np.nan)
        self.binAbsMax.fill(np.nan)
        self.binAbsMin.fill(np.nan)

    def print_keyframes(self):
        if self.can_use_keyframes:
            if self.params.fixed_cadence_keyframes():
                print(
                    "\r *    >> KeyFrames: Fixed Cadence of one out of every {} frames".format(
                        self.params.fixed_cadence_keyframes()
                    )
                )
            elif self.params.fixed_number_keyframes():
                print(
                    "\r *    >> KeyFrames: Fixed Number of Keyframes: {}".format(
                        self.params.fixed_number_keyframes()
                    )
                )
            else:
                print("Something is wrong here in the Processor.py file")
            print(
                " *    >> Selected {} keyframes out of {} total frames".format(
                    self.n_do_frames, self.n_all_frames
                )
            )
        # else:
        #     print("\r *    >> KeyFrames: Using Every Image ")

        self.super_flush(many=10)

    # print(" *    >>Selected {} keyframes out of {} total frames".format(len(self.short_list), len(self.long_list)))

    ########################################
    ## M2: For Every File in Path, do Func##
    def do_fits_function(self, fits_path=None, in_name=None, image=True):
        """
        Processes a single FITS file:
        1. Loads the FITS image.
        2. Verifies and fixes the FITS header if necessary.
        3. Performs the designated work on the FITS data.

        Parameters:
            fits_path (str): Path to the FITS file.
            in_name (str): Input name identifier.
            image (bool): Flag indicating image processing.

        Returns:
            Any: The result of the processing work or None if processing fails.
        """
        try:
            # Step 1: Load the FITS image
            if not self.load_fits_image(fits_path, in_name=in_name):
                return None

            # Step 2: Check if processing should proceed based on keyframes
            if self.use_keyframes and self.fits_path not in self.keyframes:
                return None

            if not self.should_run():
                return None

            # Step 3: Attempt to create and verify the SunPy Map
            try:
                # with fits.open(fits_path) as hdul:
                #     # Access the header from the primary HDU
                #     header = hdul[0].header

                #     # Verify and fix any issues with the header
                #     header.verify("fix")

                self.raw_map = sunpy.map.Map(fits_path)
                if isinstance(self.raw_map, list):
                    self.raw_map = self.raw_map[-1]

                # self.raw_map = sunpy.map.Map(fits_path)  # [-1]
                # self.raw_map.validate()
                # self.raw_map.verify("fix")

            except Exception as e:  # (MetaDataMissingError, MetaDataParseError):
                print("thing: " + str(e))
                # Attempt to fix the FITS header
                if self.fix_fits_header(fits_path):
                    self.fixed_fits_count += 1  # Increment fixed files counter
                    try:
                        # Retry creating and verifying the SunPy Map after fixing
                        self.raw_map = sunpy.map.Map(
                            (self.params.raw_image, self.params.header)
                        )
                        # self.raw_map.verify("silentfix")
                    except (VerifyError, OSError):
                        # If verification still fails, record the failure and skip processing
                        self.failed_fits.append(fits_path)
                        return None
                else:
                    # If fixing fails, record the failure and skip processing
                    self.failed_fits.append(fits_path)
                    return None

            # Step 4: Perform the designated processing work
            out = self.do_work()
            return out
        except StopIteration as e:
            raise e
        except Exception as e:
            raise e
            # Catch-all for any unexpected errors; optionally, you can log these
            self.failed_fits.append(fits_path)
            return None

    @staticmethod
    def fix_fits_header(fits_path):
        """
        Attempts to fix the FITS header by verifying and correcting it.

        Parameters:
            fits_path (str): Path to the FITS file.

        Returns:
            bool: True if fixing was successful, False otherwise.
        """
        try:
            with fits.open(fits_path, mode="update") as hdulist:
                hdulist.verify("fix")
                hdulist.flush()
            return True
        except Exception:
            # Suppress exceptions to maintain performance; alternatively, handle specific exceptions
            return False

    def tic(self, loud=False):
        self.tic_active = True
        self.duration = 0.0
        if self.loud_tic or loud:
            print(
                "\n\n *    Running Filter on {}...".format(self.params.current_wave()),
                end="\n",
            )
        self.tm = time.time()
        # print(f"tic {self}, tick_active = {self.tic_active}")

    def toc(self, loud=False):
        if self.tic_active:
            dur = time.time() - self.tm
            self.tm = time.time()
            if self.loud_tic or loud:
                print(
                    " ^    Done! Took: {:0.2f} seconds, or {:0.2f} mins".format(
                        dur, dur / 60
                    )
                )
            self.duration = dur
            self.params.durList.append(dur)
        else:
            pass
            # self.duration = "-1"
            # self.params.durList.append("-1")
        # print(f"toc {self}, tick_active = {self.tic_active}, dur = {self.duration}\n\n\n")
        self.tic_active = False

        # TODO make the duration list thing work, displaying the time it took to do each thing at the end.

    def should_run(self):
        return True

    def do_work(self):
        raise NotImplementedError

    def do_img_function(self):
        raise NotImplementedError

    def cleanup(self):
        self.toc()
        self.print_success()
        self.params.processors_ran.append(self)

        pass

    def setup(self):
        pass

    def process(self, params=None):
        """Load the parameters and run the algorithm"""
        self.params = params or self.params
        # import pdb; pdb.set_trace()
        if self.params is not None:
            if self.params.do_single:
                self.setup()
                self.load(self.params, quietly=False)
                mod = self.modify_one_image()
                if mod is None:
                    print(
                        " ^     No Fits Frame Saved!  ------------------------------------------------  ^\n"
                    )
            elif self.do_png:
                self.load(self.params, quietly=False)
                self.process_img_series()
            else:
                self.load(self.params, quietly=False)
                self.process_fits_series()
        # self.cleanup()

    ##  Run on Fits Files
    def process_fits_series(self):
        """Apply the function to all necessary fits files"""
        n_fits_path = len(self.keyframes)
        self.skipped = 0
        try:
            if n_fits_path > 0:
                self.setup()
                parallel = self.params.do_parallel and self.can_do_parallel
                if parallel:
                    self.parallel_fits_series()
                else:
                    self.serial_fits_series()
        except StopIteration:
            return

    def print_success(self):
        if not self.do_print_success:
            return
        try:
            n_success = self.ii + 1 - self.skipped
            if n_success + self.skipped >= 1:
                if n_success <= 0:
                    print(
                        "\r X x X-- Skipped all {} Files --xXxXxXxXxXxXxXxXxXxXxX \n".format(
                            self.skipped
                        )
                    )
                else:
                    # print(self)
                    print(
                        "\r ^ ^ ^Successfully {} {} Files ({} skipped) in {:0.4} seconds".format(
                            self.finished_verb,
                            max(n_success, 0),
                            self.skipped,
                            self.duration,
                        ),
                        flush=True,
                    )
                    print(
                        " ^ ---------------------------------------------------------------  ^\n\n"
                    )
                sleep(1)
            else:
                print(" ^    No Files Found\n")
        except ValueError as e:
            print(e)

    def serial_fits_series(self):
        # print("Running in Serial Mode...", flush=True)
        pbar = self.init_pbar_now()
        sys.stdout.flush()
        for self.ii, fits_path in enumerate(pbar):
            result = self.modify_one_fits(fits_path)
            if result is None:
                self.skipped += 1
        # print("Finished", flush=True)

    def parallel_fits_series(self):
        # print("Running in Parallel Mode...", end="")
        self.init_pool_if_needed()
        try:
            iter = self.params.multi_pool.imap_unordered(
                self.modify_one_fits, self.keyframes
            )

            pbar = self.init_pbar_now()
            for self.ii, result in enumerate(iter):
                pbar.update()
                if result is None:
                    self.skipped += 1
            # print("Finished", flush=True)
        except PicklingError as e:
            print("Parallel Run Failed: ", e)
            self.serial_fits_series()
        except (TypeError, ValueError) as e:
            self.skipped += 1

    def init_pbar_now(self, position=0):
        pbar = tqdm(
            self.keyframes,
            unit=self.progress_unit,
            desc=self.progress_string,
            position=position,
            leave=True,
        )
        return pbar

    def init_pool_if_needed(self):
        try:
            self.params.multi_pool.imap_unordered
        except AttributeError:
            # print("Using default number of cores")
            self.params.init_pool()

    @staticmethod
    def confirm_fits_file(fits_path) -> bool:
        if fits_path is not None:
            if os.path.exists(fits_path):
                if fits_path.endswith(".fits"):
                    return fits_path
                else:
                    fits_path = fits_path.replace(".jpg", ".fits")
                    return fits_path
        else:
            raise FileNotFoundError

    def modify_one_fits(self, fits_path):
        """Apply the given funtion to the given fits path"""
        fits_path = self.confirm_fits_file(fits_path)

        # self.load()
        self.in_name = self.in_name or self.params.master_frame_list_newest
        # return True
        try:

            output = self.do_fits_function(fits_path, self.in_name)
            # output=None
            self.ii += 1
        except np.linalg.LinAlgError as e:
            print("Legacy_QRN_Kernal one fits :: ", e, "\n")
            output = 0.5 * np.ones_like(self.params.raw_image)
            output[0] = 0.0
            output[1] = 1.0
        try:
            frame = output.get()
        except AttributeError as e:
            # print(e)
            frame = output

        use_name = None
        if self.frame_name is not None:
            if "mgn_rhe" in self.frame_name:
                use_name = self.frame_name
            else:
                use_name = self.out_name
        self.save_frame(frame, fits_path, self.frame_name)
        return frame

    def save_frame(self, frame, fits_path, out_name=None, force=False):
        # import pdb; pdb.set_trace()
        if frame is not None and frame is not False:
            if self.save_to_fits or force:
                # import pdb; pdb.set_trace()
                self.save_frame_to_fits_file(
                    fits_path, frame, out_name, dtype=self.out_dtype
                )

    def select_single_image(self):
        self.load_fits_paths()
        path1 = self.params.use_image_path()
        path2 = self.fits_path
        all_fits_paths = self.all_file_paths
        try:
            wavestr = self.params.current_wave()[1:]
            if "94" in wavestr:
                wavestr = wavestr[1:]
        except TypeError:
            wavestr = str(self.params.current_wave())
        wave_paths = [x for x in all_fits_paths if wavestr in x]
        if len(wave_paths) >= 1:
            self.img_path = wave_paths[0]
        elif path1 is not None or path2 is not None:
            self.img_path = path1 or path2
        else:
            raise FileNotFoundError("No frame found for wavelength: {}".format(wavestr))

    def modify_one_image(
        self,
    ):
        """Apply the given funtion to the given fits path"""

        try:
            self.select_single_image()
            self.params.modified_image = self.modify_one_fits(self.img_path)

        except NotImplementedError as e:
            self.params.modified_image = self.do_img_function()
            # self.out_name = self.out_name
        return self.params.modified_image

    def process_img_series(self):
        """Apply the function to all necessary img files"""
        self.do_one_wave(self.params.current_wave())

        # self.process_all_wavelengths()

    def process_all_wavelengths(self):
        """Run the process on all of the all_wavelengths"""
        # print(self.filt_name + ">>>", flush=True)

        folders = self.get_folders()
        for wave in folders:
            self.do_one_wave(wave)

    def do_one_wave(self, wave):
        if wave in self.params.waves_to_do:
            self.load(wave=wave)
            if len(self.params.local_imgs_paths()) > 0:
                self.process_one_wavelength(wave)

    def get_folders(self):
        base = self.params.base_directory()
        bName = self.params.batch_name()
        folders = listdir(base)
        if "fits" in folders:
            folders = listdir(dirname(base))
        elif bName in folders:
            folders = listdir(join(base, bName))
        return folders

    def process_one_wavelength(self, wave):
        raise NotImplementedError()

    def find_limb_radius(self):
        spread = 0.02
        self.limb_radius_from_fit_shrunken = self.limb_radius_from_header_shrunken
        self.limb_radius_from_fit_shrunken_forpoints = (
            self.limb_radius_from_header_shrunken_forpoints
        )
        self.lCut = int(
            self.limb_radius_from_header_shrunken - spread * self.params.rez
        )
        self.hCut = int(
            self.limb_radius_from_header_shrunken + spread * self.params.rez
        )
        return

        # print("\n", self.limb_radius_from_header_shrunken, self.limb_radius_from_fit_shrunken)
        # return

        # if self.limb_radius_already_found:
        #     # print("Had~~~~~~~~~~~~~~")
        #     return self.limb_radius_from_fit_shrunken

        # print("Needed ~~~~~~~~~~~~~~")
        # if self.limb_radius_from_fit_shrunken is not None:
        #     self.limb_radius_from_fit_shrunken = self.limb_radius_from_header_shrunken = self.limb_radius_from_fit_shrunken
        #     self.lCut = int(self.limb_radius_from_fit_shrunken - spread * self.params.rez)
        #     self.hCut = int(self.limb_radius_from_fit_shrunken + spread * self.params.rez)
        #     return self.limb_radius_from_fit_shrunken

        # if self.outer_max is None:
        #     self.load_curves(verb=False)

        # self.limb_radius_from_header_shrunken = self.params.limb_radius_from_header or 1600
        # self.limb_radius_from_header_shrunken = self.limb_radius_from_header_shrunken // self.binfactor #// self.shrink_F
        # self.limb_radius_from_header   = self.limb_radius_from_header // self.shrink_F // self.binfactor

        self.limb_radius_from_fit_shrunken = self.limb_radius_from_header_shrunken
        self.lCut = int(
            self.limb_radius_from_header_shrunken - spread * self.params.rez
        )
        self.hCut = int(
            self.limb_radius_from_header_shrunken + spread * self.params.rez
        )
        return

        try:
            do_on_running = False
            if do_on_running and self.outer_max is not None:
                outer_mid_max = self.outer_max[self.lCut : self.hCut]
                inner_mid_max = self.inner_max[self.lCut : self.hCut]
                inner_mid_min = self.inner_min[self.lCut : self.hCut]
                outer_mid_min = self.outer_min[self.lCut : self.hCut]

                outer_mid_max_maxInd = np.argmax(outer_mid_max) + self.lCut
                inner_mid_max_maxInd = np.argmax(inner_mid_max) + self.lCut
                inner_mid_min_maxInd = np.argmax(inner_mid_min) + self.lCut
                outer_mid_min_maxInd = np.argmax(outer_mid_min) + self.lCut

                self.peak_indList = [
                    outer_mid_max_maxInd,
                    inner_mid_max_maxInd,
                    inner_mid_min_maxInd,
                    outer_mid_min_maxInd,
                ]
            else:
                max_curve = self.frame_maximum[self.lCut : self.hCut]
                min_curve = self.frame_minimum[self.lCut : self.hCut]
                max_ind = np.argmax(max_curve) + self.lCut
                min_ind = np.argmax(min_curve) + self.lCut
                self.peak_indList = [max_ind, min_ind]

            self.limb_radius_from_fit_shrunken = np.round(np.mean(self.peak_indList), 6)
        except TypeError as e:
            # print("\r        find_limb_radius failed: ", e)
            self.limb_radius_from_fit_shrunken = self.limb_radius_from_header_shrunken

        spread = 0.005
        self.lCut = int(self.limb_radius_from_fit_shrunken - spread * self.params.rez)
        self.hCut = int(self.limb_radius_from_fit_shrunken + spread * self.params.rez)
        self.limb_radius_already_found = True

    def init_radius_array(
        self, vignette_radius=1.51, s_radius=400, t_factor=1.28, force=False
    ):
        """Build an r-coordinate array of shape(in_object)"""
        # self.params.rez = self.params.modified_image.shape[0]
        self.init_image_frames()
        self.determine_shrink_factor()
        self.make_radius()
        self.find_limb_radius()
        # self.make_vignette(vignette_radius)
        if True:  # type(self) is QRNProcessor:
            self.init_bin_array()

    def double_smash(self, raw_arr, log=True, prerun=True):
        if prerun:
            tmin, tmax = np.nanpercentile(raw_arr, [0.1, 99.9])
            flat_norm = (raw_arr - tmin) / (tmax - tmin)
        else:
            flat_norm = raw_arr
        flat_arr = np.log10(flat_norm) if log else flat_norm
        is_finite = flat_arr[np.isfinite(flat_arr)]
        tmin, tmax = np.nanpercentile(is_finite, [0.1, 99.99])
        flat_arr_norm = (flat_arr - tmin) / (tmax - tmin)
        flat_arr_norm[~np.isfinite(flat_arr)] = np.nan
        return flat_arr_norm

    def resize_image(self, img=None, want_rez=1024, prnt=True):
        # if prnt: print("   * Shrinking Rez to {}...".format(want_rez))
        # if self.params.second_shape == want_rez:
        #     return self.params.raw_image

        img = img if img is not None else self.params.raw_image
        first_shape = img.shape[0]
        from sunback.utils.array_util import reduce_array

        self.params.raw_image, self.params.center, self.shrink_F = reduce_array(
            self.params.raw_image, self.params.center, want_rez
        )
        self.params.modified_image, _, _ = reduce_array(
            self.params.modified_image, self.params.center, want_rez
        )
        self.params.rez = want_rez

        # if self.params.modified_image is not None and self.params.modified_image.shape != self.params.raw_image.shape:
        # self.header["NAXIS1"] = want_rez
        # second_shape = self.params.raw_image.shape[0]
        # self.shrink_F = first_shape // second_shape #4 if second_shape == 1024 else 2 if second_shape == 2048 else 1
        # self.init_image_frames()
        # self.parse_resize_args(self.shrink_F)
        # self.make_radius()
        # self.make_vignette()
        self.smol = True
        return self.params.raw_image

    def ensure_odd(self, number):
        number = int(np.round(number))
        if ~number % 2:
            number += 1
        return number

    def init_image_frames(self):
        mdi = self.params.modified_image
        do = False
        if mdi is None:
            do = True
        if type(mdi) in [list, tuple]:
            if len(mdi) == 0:
                do = True
        if type(mdi) == np.ndarray:
            if mdi.size == 1:
                do = True
        if do:
            mdi = np.float16(self.params.raw_image)
            self.params.modified_image = mdi

        return self.params.modified_image

    def determine_shrink_factor(self):
        # self.binfactor = 4
        # = self.header["NAXIS1"]
        self.params.rez = rez = self.params.rez or self.header["NAXIS1"]

        if rez == 4096:
            self.shrink_factor = 1
        elif rez == 2048:
            self.shrink_factor = 2
        elif rez == 1024:
            self.shrink_factor = 4
        else:
            self.shrunk_factor = 1
            # raise NotImplementedError

        self.parse_shrink_args()
        self.find_limb_radius()

    def parse_shrink_args(self, shrink_needed=True):
        nn = self.shrink_factor if shrink_needed else 1
        # if self.limb_radius_from_header_shrunken is None:

        x0 = self.header.get("X0_MP", self.params.rez // 2)
        y0 = self.header.get("Y0_MP", self.params.rez // 2)
        rsun = self.header.get("R_SUN", self.params.rez // 2)

        self.params.center_NOTforpoints = [x0 / (nn), y0 / (nn)]
        self.limb_radius_from_header = rsun
        self.limb_radius_from_header_shrunken = rsun / nn
        self.output_abscissa = np.arange(self.params.rez)

        self.params.center = [x0 / (nn), y0 / (nn)]
        self.limb_radius_from_header_shrunken_forpoints = rsun / nn
        self.ratio_factor_for_radius = (
            self.limb_radius_from_header_shrunken_forpoints
            / self.limb_radius_from_header_shrunken
        )
        # print("\n", self.params.center, self.params.center_NOTforpoints)
        # print(self.limb_radius_from_header_shrunken_forpoints, self.limb_radius_from_header_shrunken)
        # a=1

        # print(self.limb_radius_from_header_shrunken)

    # def parse_resize_args(self, factor=4):
    #     if self.limb_radius_from_header_shrunken is None:
    #         self.params.center = [self.header["X0_MP"] / (factor), self.header["Y0_MP"] / (factor)]
    #         self.limb_radius_from_header_shrunken = self.header["R_SUN"] / factor
    #         self.output_abscissa = np.arange(self.params.rez)
    # print(self.limb_radius_from_header_shrunken)

    def make_radius(self):
        self.xx, self.yy = np.meshgrid(
            np.arange(self.params.rez), np.arange(self.params.rez)
        )
        # if self.frame_name == "lev1p5":
        #     self.params.center = [self.params.rez//2, self.params.rez//2]
        self.xc, self.yc = xc, yc = (
            self.xx - self.params.center_NOTforpoints[0],
            self.yy - self.params.center_NOTforpoints[1],
        )
        self.radius = np.sqrt(xc * xc + yc * yc)
        self.theta_array = np.arctan2(yc, xc)
        self.rad_flat = self.radius.flatten() + 0

        self.xcp, self.ycp = xcp, ycp = (
            self.xx - self.params.center[0],
            self.yy - self.params.center[1],
        )
        self.radius_forpoints = np.sqrt(xcp * xcp + ycp * ycp)
        self.theta_array_forpoints = np.arctan2(ycp, xcp)
        self.rad_flat_forpoints = self.radius_forpoints.flatten() + 0

    def make_vignette(self, vignette_radius=1.51):
        self.vig_radius_pix = self.r2n(vignette_radius)
        self.vig_radius_rr = self.n2r(self.vig_radius_pix)
        # self.detector_radius_rr = self.n2r(self.params.rez // 2)

        self.vignette_mask = np.asarray(
            self.radius > self.vig_radius_pix * 2, dtype=bool
        )

    def init_bin_array(self):
        self.binInds = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        self.binInds_forpoints = np.asarray(
            np.floor(self.rad_flat_forpoints), dtype=np.int32
        )
        # self.binXX   = self.xx.flatten()
        # self.binYY   = self.yy.flatten()
        self.binII = np.arange(len(self.rad_flat))
        self.binRR = np.round(self.rad_flat / self.limb_radius_from_header_shrunken, 4)

    @staticmethod
    def get_bin_items(bin_list):
        """Retrieve finite values from a bin_list"""
        bin_array = np.asarray(bin_list)
        finite = np.isfinite(bin_array)
        filled = bin_array != 0
        keep = list(np.nonzero(finite & filled)[0])
        finite_out = bin_array[keep]
        return keep, finite_out

    def bin_radially(self):  # TODO Make the save to fits work
        """Bin the intensities by radius"""
        self.do_binning(fast=False)
        # do_cache = False
        # if do_cache:
        #     if not self.there_is_cached_data:
        #         self.do_binning(fast=False)
        #         self.save_cached_data(self.radBins)
        #         self.there_is_cached_data = True
        #     else:
        #         self.load_cached_data(self.radBins)
        # else:
        #     self.do_binning(fast=False)

    def initialize_binning(self, use_im, binBoxSize):
        flat_im = self.params.modified_image if use_im is None else use_im
        self.orig_size = flat_im.shape
        self.params.rez = flat_im.shape[0]
        flat_im = flat_im.flatten()
        sz = (self.params.rez, self.params.rez)
        self.params.rhe_image = np.empty(self.orig_size).flatten()
        self.params.rhe_image.fill(np.nan)
        self.n_inds = np.max(self.binInds)
        self.equal_intensity_array = np.empty((self.n_inds, binBoxSize))
        self.equal_radius_array = np.empty((self.n_inds, binBoxSize))
        self.equal_mean_array = np.empty((self.n_inds))
        self.equal_std_array = np.empty((self.n_inds))

        self.equal_intensity_array.fill(np.nan)
        self.equal_radius_array.fill(np.nan)
        self.equal_mean_array.fill(np.nan)
        self.equal_std_array.fill(np.nan)
        return flat_im

    def do_binning(
        self, use_im=None, fast=False, binBoxSize=100
    ):  # Bin the intensities by radius
        flat_im = self.initialize_binning(use_im, binBoxSize)
        if False:
            params_list = tqdm(
                np.arange(self.n_inds),
                desc=" *    Sorting Pixels",
                position=0,
                leave=True,
            )
        else:
            params_list = np.arange(self.n_inds)
        if self.out_name:
            skip = 1 if "RHE" in self.out_name else 15 if fast else 1
        else:
            skip = 1

        for binI in params_list:
            if not np.mod(binI, skip):
                if fast:
                    self.fast_binning(binI, flat_im, binBoxSize)
                else:
                    self.full_binning(binI, flat_im, skip=skip)

        if False:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", sharey="all")
            fig.set_size_inches(5, 8)
            ax1.imshow(
                np.sqrt(self.params.raw_image),
                interpolation=None,
                origin="lower",
                cmap="viridis",
            )
            ax2.imshow(
                self.params.rhe_image.reshape(self.orig_size),
                interpolation=None,
                origin="lower",
                cmap="viridis",
            )
            ax2.scatter(*self.params.center, c="r", s=10)
            ax1.scatter(*self.params.center, c="r", s=10)
            ax1.set_title("Original")
            ax2.set_title("Radial Histogram Equalization")
            fig.suptitle("Eclipse Photo on Pixel 6")
            plt.tight_layout()
            plt.show()

        return self.equal_radius_array, self.equal_intensity_array

    def fast_binning(self, binI, flat_im, binBoxSize):
        entries, the_mean, the_std = self.get_bin_entries(binI, flat_im)
        limit = entries.shape[0]
        if limit:
            indices = np.random.choice(limit, binBoxSize, replace=True)
            (
                (
                    good_coord,
                    self.equal_intensity_array[binI, :],
                    self.equal_radius_array[binI, :],
                ),
                self.equal_mean_array[binI],
                self.equal_std_array[binI],
            ) = entries[indices].T, the_mean, the_std

    @staticmethod
    def squashfunc(array):
        # return array
        return np.sqrt(array + 0)
        # return np.log10(array)

    def autoLabelPanels(self, axArray, loc=(0.045, 0.05), messages=None, color="r"):
        for ii, ax in enumerate(axArray.flatten()):
            message = "" if messages is None else messages[ii]
            ax.annotate(
                "({})  {}".format(chr(97 + ii), message),
                loc,
                color=color,
                xycoords="axes fraction",
            )

    def safe_indexing(self, array, indices):
        # Create an output array filled with NaN values
        output = np.full(len(indices), np.nan)

        # Filter indices that are within the valid range
        valid_indices = indices[(indices >= 0) & (indices < len(array))]

        # Set valid values into the output array
        output[(indices >= 0) & (indices < len(array))] = array[valid_indices]

        return output

    def safe_indexing_update(self, array, indices, values):
        # Ensure indices are within valid range
        valid_mask = (indices >= 0) & (indices < len(array))
        valid_indices = indices[valid_mask]
        valid_values = values[valid_mask]

        # Update the array at the valid indices
        array[valid_indices] = valid_values

    def full_binning(self, binI, image, skip=1):
        entries, mean, std = self.get_bin_entries(binI, image)
        if entries.size == 0:
            # print("No valid entries found.")
            return

        (good_coord, bin_array, radii) = entries.T

        best_coords = np.floor(good_coord).astype(int)
        best_coords = np.clip(
            best_coords, 0, len(self.params.rhe_image) - 1
        )  # Ensure within bounds
        ranks = stats.rankdata(bin_array, "average") / len(bin_array)

        # Use safe_indexing_update to assign ranks safely to self.params.rhe_image
        self.safe_indexing_update(self.params.rhe_image, best_coords, ranks)

    def get_bin_entries(self, binI, image=None):
        the_inds = np.where(self.binInds == binI)
        the_inds_forpoints = np.where(self.binInds_forpoints == binI)

        # Flattening the image for simpler indexing
        flat_image = image.flatten() if image is not None else self.flat_im

        # Using safe indexing to handle out-of-bound indices
        bin_array = self.safe_indexing(flat_image, the_inds_forpoints[0])
        coords = self.safe_indexing(self.binII, the_inds_forpoints[0]).astype(int)

        # Filter out NaN entries which indicate out-of-bound indices
        valid_entries = ~np.isnan(bin_array)
        good_coord = coords[valid_entries]
        bin_array = bin_array[valid_entries]

        # Calculating radii for valid coordinates only
        radii = [self.binRR[x] for x in good_coord]

        return (
            np.asarray([good_coord, bin_array, radii]).T,
            np.nanmean(bin_array),
            np.nanstd(bin_array),
        )

    # def full_binning(self, binI, image, skip=1):
    #     try:
    #         entries, mean, std = self.get_bin_entries(binI, image)
    #     except IndexError as e:
    #         print(e, "Bin # ", binI)
    #     (good_coord, bin_array, radii) = entries.T
    #     if len(bin_array) > 0:
    #         # self.binBox.append(np.asarray([good_coord, radii, bin_array]).T.tolist())
    #         # from src.processor.QRNProcessor import QRNpreProcessor
    #         if "qrn" in str(type(self)).casefold():
    #             # use_percentiles = [98.5, 90, 7, 4]
    #             # use_percentiles = [99, 95, 5, 1]
    #             use_percentiles = [99, 99.5, 4, 1]
    #             # A,B,C,D =
    #             array = np.arange(binI, np.min((binI+skip, self.bin_rez)))
    #             self.binAbsMax[array], self.binMax[array], self.binMin[array], self.binAbsMin[array] = np.nanpercentile(bin_array, use_percentiles)
    #         else:
    #             # best_coords = np.asarray([x for x in good_coord if x < len(image.flatten())]).astype(int)
    #                 best_coords = np.floor(good_coord).astype(int)
    #                 ranks = stats.rankdata(bin_array, "average") / len(bin_array)
    #                 best_coords = np.clip(best_coords, 0, len(image) - 1)  # Clip the value to prevent index errors
    #                 self.params.rhe_image[best_coords] = ranks #This is RHE
    #     return good_coord, radii, bin_array

    # def get_bin_entries(self, binI, image=None):
    #     # frame = self.flat_im if frame is None else frame

    #     # want_radius =       self.binRR[binI]
    #     # the_inds =          np.where(self.binRR == want_radius)

    #     the_inds =          np.where(self.binInds == binI)
    #     the_inds_forpoints = np.where(self.binInds_forpoints == binI)
    #     # the_inds_forpoints = [x for x in the_inds_forpoints if x < len(image.flatten())]
    #     # print("it is the same : ", np.all(the_inds[0] == the_inds_forpoints[0]))

    #     keep, bin_array =   self.get_bin_items(image[the_inds_forpoints])
    #     coord =             self.binII[the_inds_forpoints].tolist()
    #     good_coord =        [coord[x] for x in keep]
    #     radii = [self.binRR[int(x)] for x in good_coord]
    #     # radii = [self.binInds[int(x)] for x in good_coord]
    #     return np.asarray([good_coord, bin_array, radii]).T, np.mean(bin_array), np.std(bin_array)

    def mask_out_sun(self, image, radius=None, mask=None, plug=None, radius2=0.9):
        if self.radius is None:
            self.init_radius_array()

        if radius is None:
            radius = 1.01
        mask = mask or np.nan if "f" in str(image.dtype) else 0

        if len(image.shape) > 2:
            image[:, self.radius / self.limb_radius_from_fit_shrunken < radius] = mask
            if plug is not None:
                image[:, self.radius / self.limb_radius_from_fit_shrunken < radius2] = (
                    plug
                )

        else:
            image[self.radius / self.limb_radius_from_fit_shrunken < radius] = mask
            if plug is not None:
                image[self.radius / self.limb_radius_from_fit_shrunken < radius2] = plug
        return image

    def get_even_points_in_radius(
        self, binBoxSize=100, image=None
    ):  # equally spaced points
        # Get an even number of items vs radius
        binRad = []
        binInts = []

        self.do_binning(use_im=image, fast=True, binBoxSize=binBoxSize)
        return self.equal_radius_array, self.equal_intensity_array

        # if use_image is not None:
        #     self.binBox = []
        #
        # elif self.binBox is None:
        #     self.binBox = []
        #     self.do_binning(use_im=use_image, fast=True)
        #
        # for box in self.binBox:
        #     try:
        #         subset = choices(box, k=binBoxSize)
        #         for (radInd, radius, intensity) in subset:
        #             binRad.append(radius)
        #             binInts.append(intensity)
        #     except ValueError as e:
        #         print(e)
        # binRad, binInts = np.asarray(binRad), np.asarray(binInts)
        # # choices(tup, k=self.binBoxSize)
        # return binRad, binInts

    def do_compare_histogramplot(
        self, frames=None, names=None, even_points=100, use_cmap=False
    ):
        # self.prep_histograms()
        frames = (
            frames
            if frames is not None
            else [
                self.params.raw_image2.reshape(self.params.modified_image.shape),
                self.params.modified_image.reshape(self.params.modified_image.shape),
                self.params.rhe_image.reshape(self.params.modified_image.shape),
            ]
        )

        names = (
            names
            if names is not None
            else ["Log10 (Normalized)", "QRN (Normalized)", "RHE"]
        )

        fig, axArray = plt.subplots(
            3,
            len(frames),
            sharex="row",
            sharey="row",
            gridspec_kw={"height_ratios": [2, 1.5, 2.5]},
        )
        (top_axes, mid_axes, bot_axes) = axArray
        try:
            t_rec = self.header["T_REC"]
        except KeyError as e:
            t_rec = self.header["T_OBS"]
        fig.suptitle("{}  at  {}".format(self.wave, t_rec))

        # import copy
        # frames2 = copy.deepcopy(frames)
        self.plot_histogram_images(top_axes, frames, names)
        self.plot_histogram_points(bot_axes, frames, names, even_points, axes2=mid_axes)

        mid_axes[0].legend(frameon=False, fontsize=7)
        bot_axes[0].set_ylabel("Intensity")
        bot_axes[1].legend(frameon=False, fontsize=7)
        fig.set_size_inches((16, 8))
        # plt.tight_layout()
        plt.subplots_adjust(
            top=0.92, bottom=0.073, left=0.04, right=0.985, hspace=0.218, wspace=0.1
        )
        # print("HELLO WORLD")
        # plt.savefig(
        #     os.path.expanduser(
        #         r"~/vscode/sunback_data/renders/Single_Test/imgs/mod/histograms_all_hq.png",
        #     ),
        #     dpi=600,
        # )
        # plt.savefig(
        #     os.path.expanduser(
        #         r"~/vscode/sunback_data/renders/Single_Test/imgs/mod/histograms_all_lq.png",
        #     ),
        #     dpi=400,
        # )
        plt.savefig(
            os.path.expanduser(
                f"~/vscode/sunback_data/renders/{self.params.batch_name()}/imgs/mod/histograms_all_vlq.pdf",
            ),
            dpi=300,
        )
        plt.close(fig)
        # plt.savefig(r"G:\sunback_images\Single_Test\imgs\histograms.pdf", dpi=400)
        # plt.show()
        # self.maximizePlot()
        # plt.tight_layout()
        # plt.show(block=True)
        # asdf = 1

    def do_compare_histogramplot_rheonly(
        self,
        frames=None,
        names=None,
        target_names=None,
        even_points=150,
        use_cmap=False,
    ):
        # If target_names are provided, select only those frames with exact matching names
        if target_names:
            # Find the indices of the frames that match the target_names exactly
            has_rhe = [i for i, nam in enumerate(names) if nam in target_names]
        else:
            # Fallback to the original behavior if no target_names are provided
            has_rhe = np.where(
                ["rhe" in nam or "comp" in nam.casefold() for nam in names]
            )[0]

        # Select the corresponding frames and names using the indices
        framesq = [frames[int(x)] for x in has_rhe]
        namesq = [names[int(x)] for x in has_rhe]
        # print("targets: ", framesq)
        # print("targets: ", namesq)

        # Select specific frames and names (this part of the logic stays unchanged)
        fram = [*framesq]
        name = [*namesq]

        fig, axArray = plt.subplots(
            2,
            len(fram),
            sharex="row",
            sharey="row",
            gridspec_kw={"height_ratios": [3, 2]},
        )
        (top_axes, bot_axes) = axArray
        try:
            t_rec = self.header["T_REC"]
        except KeyError as e:
            t_rec = self.header["T_OBS"]
        fig.suptitle("{}  at  {}".format(self.wave, t_rec))

        self.params.cmap = self.cmap = aia_color_table(
            int(self.wave) * u.angstrom
        )  # frames2 = copy.deepcopy(frames)

        self.plot_histogram_images(top_axes, fram, name)
        self.plot_histogram_points(bot_axes, fram, name, even_points, axes2=None)

        # mid_axes[0].legend(frameon=False)
        bot_axes[0].set_ylabel("Pixel Value")
        bot_axes[0].legend(frameon=False)  # , loc='lower left')
        fig.set_size_inches((12, 8))
        plt.tight_layout()
        fig.subplots_adjust(top=0.93)
        print("I'M PLOTTING")
        plt.savefig(
            os.path.expanduser(
                r"~/vscode/sunback_data/renders/Single_Test/imgs/mod/histograms_rhe2_hq.png"
            ),
            dpi=600,
        )

        plt.savefig(
            os.path.expanduser(
                r"~/vscode/sunback_data/renders/Single_Test/imgs/mod/histograms_rhe2_lq.png"
            ),
            dpi=400,
        )
        plt.savefig(
            os.path.expanduser(
                r"~/vscode/sunback_data/renders/Single_Test/imgs/mod/histograms_rhe2_vlq.png"
            ),
            dpi=300,
        )
        print("Plotting Complete")
        # plt.savefig(r"G:\sunback_images\Single_Test\imgs\histograms_rhe.png", dpi=400)
        # plt.savefig(r"G:\sunback_images\Single_Test\imgs\histograms.pdf", dpi=400)
        # plt.show()
        # self.maximizePlot()
        # plt.tight_layout()
        plt.close(fig)
        # plt.show(block=True)
        # asdf = 1

    def do_compare_histogramplot_images(
        self, frames=None, names=None, even_points=150, use_cmap=False
    ):
        has_rhe = np.where(["rhe" in nam for nam in names])[0]
        framesq = [frames[int(x)] for x in has_rhe]
        namesq = [names[int(x)] for x in has_rhe]
        namesqq = [names[int(x)] + "_raw" for x in has_rhe]

        fram = frames[2:]  # [framesq[0], framesq[1], framesq[1]]
        name = names[2:]  # [namesq[0], namesqq[1], namesq[1]]

        fig, axArray = plt.subplots(4, len(fram), sharex="row", sharey="row")
        (top_axes, mid_axes, low_mid_axes, bot_axes) = axArray
        try:
            t_rec = self.header["T_REC"]
        except KeyError as e:
            t_rec = self.header["T_OBS"]

        # import copy
        # frames2 = copy.deepcopy(frames)
        nan_names = [None for x in names]
        self.plot_histogram_images(top_axes, fram, name)
        self.plot_histogram_images(mid_axes, fram, nan_names)
        self.plot_histogram_images(low_mid_axes, fram, nan_names)
        self.plot_histogram_images(bot_axes, fram, nan_names)
        # self.plot_histogram_points(bot_axes, fram, name, even_points, axes2=None)
        fig.set_size_inches((16, 10))

        top_axes[0].set_xlim((1393, 2703))
        top_axes[0].set_ylim((0, 1000))

        mid_axes[0].set_xlim((2200, 4096))
        mid_axes[0].set_ylim((1612, 3013))

        low_mid_axes[0].set_xlim((3013, 4096))
        low_mid_axes[0].set_ylim((650, 1426))

        bot_axes[0].set_xlim((0, 1200))
        bot_axes[0].set_ylim((477, 1413))
        # mid_axes[0].legend(frameon=False)
        # bot_axes[0].set_ylabel("Intensity")
        # bot_axes[0].legend(frameon=False) #, loc='lower left')
        fig.suptitle("{}  at  {}".format(self.wave, t_rec))
        plt.tight_layout()
        plt.tight_layout()
        plt.subplots_adjust(
            wspace=0,
            hspace=0,
            top=0.97,
        )
        pth = f"~/vscode/sunback_data/renders/{self.batch_name}/imgs/mod"
        os.makedirs(pth, exist_ok=True)

        plt.savefig(
            os.path.expanduser(
                os.path.join(pth, self.wave, "/histograms_images_hq.png")
            ),
            dpi=300,
        )
        plt.savefig(
            os.path.expanduser(
                os.path.join(pth, self.wave, "/histograms_images_lq.png")
            ),
            dpi=200,
        )

        plt.close(fig)
        # plt.savefig(r"~/vscode/sunback_data/renders/Single_Test/imgs/mod\histograms.pdf", dpi=400)
        # plt.show()
        # self.maximizePlot()
        # plt.tight_layout()
        # plt.show(block=True)
        # asdf = 1

    def plot_histogram_images(self, axes, frames, names, donorm=True, dosmash=True):
        ## Plot Images
        print(" *    Plotting Images")
        for ax, frame, nam in zip(axes, frames, names):
            frame = self.histNorm(frame, donorm=donorm, dosmash=dosmash, name=nam)
            self.plot_one_histimage(ax, frame, title=nam)

    def plot_histogram_points(
        self, axes, frames, names, even_points, donorm=True, dosmash=True, axes2=None
    ):
        # Plot the Histograms
        print(" *    Plotting Histograms")
        if axes2 is None:
            axes2 = [None] * len(axes)
        for ax, ax2, frame, nam in zip(axes, axes2, frames, names):
            frame = self.histNorm(frame, donorm=donorm, dosmash=dosmash, name=nam)
            # ax.set_title(nam)
            self.plot_one_histogram(ax, ax2, frame, nam, even_points=even_points)
            # return

    def histNorm(
        self, frame, hi=99.0, lo=0.5, donorm=True, dosmash=True, name="default"
    ):
        skiplist = ["raw"]
        if name is not None:
            for item in skiplist:
                if item in name:
                    return frame
        if donorm:
            frame = self.normalize(frame, hi, lo)
        return frame

    def plot_one_histimage(self, ax, frame, title=None):
        sz = (
            self.params.rez // self.shrink_factor,
            self.params.rez // self.shrink_factor,
        )
        szz = int(np.round((frame.shape[0])))
        frame = frame.reshape((szz, szz))

        # if not self.cmap and self.params.wave:
        from sunpy.visualization.colormaps import color_tables as ct

        self.cmap = ct.aia_color_table(int(self.wave) * u.angstrom)

        ax.imshow(frame, origin="lower", cmap=self.cmap, vmin=0, vmax=1)
        if title is not None:
            ax.set_title(title)

        ax.set_facecolor("k")

    def plot_one_histogram(self, ax, ax2, frame, title=None, even_points=100):
        absiss, frame = self.get_even_points_in_radius(even_points, frame)
        self.plot_frame_hist(ax, ax2, frame, title, hist_absiss=absiss)
        # ax.set_title(title)

    def plot_frame_hist(
        self, ax1, ax2, use_image, title="Default", blk_alpha=0.2, hist_absiss=None
    ):
        # Gather Points to Display
        # flat_sunback = self.params.modified_image.flatten() + 0
        hist_absiss = hist_absiss if hist_absiss is not None else self.hist_absiss
        if len(use_image.shape) > 1:
            use_image = use_image.flatten()
        ax1.scatter(
            hist_absiss, use_image, c="k", s=4, alpha=blk_alpha, edgecolors="none"
        )
        lo, hi, num = (
            np.nanmin(hist_absiss),
            np.nanmax(hist_absiss),
            len(self.equal_mean_array),
        )
        absiss = np.linspace(lo, hi, num)

        inds = ~np.isnan(self.equal_mean_array) & np.isfinite(self.equal_mean_array)
        if ax2 is not None:
            ax2.plot(
                absiss[inds], self.equal_mean_array[inds], c="k", ls="-", label="Mean"
            )
            ax2.plot(
                absiss[inds],
                self.equal_std_array[inds],
                c="grey",
                ls=(0, (5, 1)),
                label="Std",
            )
            ax2.set_ylim((-0.2, 1.2))

        # Formatting the Plot
        vloc = self.n2r(self.params.rez / 2)
        do_legend = "rhe(lev1p5)" in title

        # ax1.set_title(title)
        use_axes = (ax1, ax2) if ax2 is not None else [ax1]
        for ax in use_axes:
            ax.axhline(0, c="lightgrey", ls="-")
            ax.axhline(1, c="lightgrey", ls="-")
            ax.axvline(1, c="grey", label="Solar Limb" if do_legend else None)
            ax.axvline(
                vloc, c="grey", ls=":", label="Detector Edge" if do_legend else None
            )
            ax.axvline(
                self.vig_radius_rr,
                c="lightgrey",
                ls=":",
                label="Optical Edge" if do_legend else None,
            )
            ax.set_xlim((-0.05, 1.9))
        ax1.set_ylim((-0.3, 1.5))
        ax1.set_xlabel("Distance from Sun Center")
        # plt.show(block=True)

    def prep_histograms(self):
        # self.params.rhe_image = self.params.rhe_image.reshape(self.params.modified_image.shape)

        skip_points = 10 if self.params.rez < 3000 else 300
        self.hist_absiss = self.n2r(self.rad_flat[::skip_points])
        self.hist_argsort = np.argsort(self.hist_absiss)
        self.hist_absiss_sorted = self.hist_absiss[self.hist_argsort]

    @staticmethod
    def maximizePlot():
        try:
            mng = plt.get_current_fig_manager()
            backend = plt.get_backend()
            if backend == "TkAgg":
                try:
                    mng.window.state("zoomed")
                except:
                    mng.resize(*mng.window.maxsize())
            elif backend == "wxAgg":
                mng.frame.Maximize(True)
            elif backend[:2].upper() == "QT":
                mng.window.showMaximized()
            else:
                return False
            return True
        except:
            return False

        # top_axes[ii].imshow( fram.reshape(self.params.raw_image2.shape),  origin='lower', cmap='gray', vmin=0, vmax=1)

    # self.plot_one_histogram(bot_axes[1], img2, lab2, donorm=False, even_points=even_points)
    # self.plot_one_histogram(bot_axes[2], img3, lab3, donorm=False, dosmash=False, even_points=even_points)

    # if use_cmap:
    #     self.params.cmap = aia_color_table(int(self.wave) * u.angstrom)
    # else:
    #     from matplotlib import cm
    #     self.params.cmap = cm.gray

    # if "log10" in lab1:
    #     img1 = self.orig_smasher(img1)

    ## Plot Images
    # top_axes[1].imshow( img2,  origin='lower', cmap='gray', vmin=0, vmax=1)
    # top_axes[2].imshow( img3,  origin='lower', cmap='gray', vmin=0, vmax=1)
    # goal = np.sqrt(self.binInds.max())
    # basis = np.linspace(1, goal, len(self.hist_absiss))
    # # basis = np.logspace(0,2)
    # # wantInds = basis
    # def plot_aia_changed(self):

    ########################################
    ## M3: Identify Directory of Interest ##
    ########################################

    def discover_best_root_directory(
        self, subdirectory_name="sunback_images", drive=None
    ):
        """Determine where to store the images"""
        if __file__ in globals():
            ddd = dirname(abspath(__file__))
        else:
            ddd = abspath(getcwd())

        while "dropbox".casefold() in ddd.casefold():
            ddd = abspath(join(ddd, ".."))

        directory = join(ddd, subdirectory_name)

        if drive:
            directory[0] = drive

        if not isdir(directory):
            makedirs(directory)
        return directory

    ############################
    ## M4: Save Frame to Fits ##
    ############################

    def save_frame_to_fits_file(
        self, fits_path, frame, out_name=None, dtype=None, shrink=True
    ):
        """Save a fits file to disk"""
        # print(f"\t\tSaving {out_name} to Fits File...", end="")

        # Check if the file exists
        file_exists = os.path.exists(fits_path)

        if not file_exists:
            # Create a new FITS file
            # hdul = fits.HDUList([fits.PrimaryHDU()])
            hdul = fits.HDUList([])
        else:
            # Open the existing FITS file
            hdul = fits.open(
                fits_path, mode="update", memmap=False, ignore_missing_simple=True
            )
            hdul.verify("silentfix+ignore")  # Verify the file
        # self.frame_name = "jpeg"
        try:
            if out_name is None:
                in_name = self.frame_name
                entries = in_name.split("(")
                previous_name = entries[-1].replace(")", "")
                last_name = entries[0]
                this_filters_name = str(self.out_name)
                the_original = "({})".format(last_name)
                field = this_filters_name + the_original
                field = field.casefold()
            else:
                field = out_name.casefold()

            frame2 = np.copy(
                frame
            )  # Create a copy of the frame to avoid modifying the original

            if len(frame2.shape) > 2:
                frame2 = np.sum(frame2, axis=-1)

            if "float" in str(frame.dtype):
                frame2 = frame2.astype(np.float32)

            # import pdb; pdb.set_trace()
            if self.header is None:
                self.header = fits.Header()
                self.header["IMG_TYPE"] = "LIGHT"
                self.header["EXPTIME"] = 3.0
                self.header["CUNIT1"] = "degree"
                self.header["CUNIT2"] = "degree"
                self.header["X0_MP"] = self.params.center[0]
                self.header["Y0_MP"] = self.params.center[1]
                self.header["R_SUN"] = 960 * 10**3
                self.header["T_OBS"] = "04-08-2024 12:00:00"

            fit_frame = fits.ImageHDU(frame2, name=field, header=self.header)

            if field not in hdul:
                hdul.append(fit_frame)  # Write
            else:
                hdul[field] = fit_frame  # Write
            # import pdb; pdb.set_trace()
            hdul.writeto(fits_path, output_verify="fix", overwrite=True)

            hdul.close(output_verify="fix")

            if self.params.speak_save:
                middle = " *         ** >> Saved Frame {} << **".format(field)
                midlen = len(middle) - 14
                print(" * \n *         ** " + "V" * midlen + " **")
                print(middle)
                print(" *         ** " + "^" * midlen + " **\n * ")
                # print("File Saved!")
        except PermissionError as e:
            print(
                "\n        !! No Permission to save the file: \n         {}".format(
                    fits_path
                )
            )
            self.skipped += 1
        except FileNotFoundError as e:
            print(
                "\n        !! No File to save the file: \n         {}".format(fits_path)
            )
            self.skipped += 1

    def make_shortcut(self, file_in_path=None, shortcut_out_path=None, doAppend=True):
        path = self.params.shortcut_directory(shortcut_out_path)
        # import os, winshell, win32com.client, Pythoncom
        import os, win32com.client

        basename = os.path.basename(file_in_path)
        basename = basename.replace("___raw.avi", "")
        basename = basename.replace("__comp.avi", "")
        basename = basename.replace("_small.avi", "")
        if doAppend:
            path = os.path.join(path, "{}.lnk".format(basename))
        # print(path)

        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = path
        shortcut.IconLocation = file_in_path
        shortcut.save()

    def delete_further_hdus(self, hdul, field):
        try:
            self.list_hdus(hdul)
            ii = self.hdu_name_list.index(field) + 1
            return hdul[0:ii]
        except ValueError as e:
            # print(e)
            return hdul

    def load_last_fits_field(self, fits_path):
        """Load a fits file from disk"""
        return self.load_this_fits_frame(fits_path, -1)

    def load_first_fits_field(self, fits_path):
        """Load a fits file from disk"""
        if "1600" in fits_path:
            a = 1
        fields = self.load_this_fits_frame(fits_path, 0)
        if fields[0] is None:
            fields = self.load_this_fits_frame(fits_path, 1)
        return fields

    def rename_start_frames(self, fits_path):
        with fits.open(
            fits_path,
            cache=False,
            mode="update",
            ignore_missing_end=True,
            output_verify="silentfix+ignore",
        ) as hdul:
            self.rename_initial_frames(hdul)  # This might not work

    @staticmethod
    def rename_initial_frames(hdul):
        # vprint("Blank Frame Ran")
        # for ii in [0,1]:
        #     try:
        #         level = hdul[ii].header['LVL_NUM']
        #         level2= hdul[ii].level
        #         break
        #     except:
        #         continue

        # level_string = 'lev' + str(level).replace('.', 'p')
        # a=1
        pass

        hdu_name_list = [frame.name.casefold() for frame in hdul]
        for to_replace in ["COMPRESSED_IMAGE", ""]:  # , 'PRIMARY']:
            if to_replace.casefold() in hdu_name_list:
                for item in hdul:
                    if item.name.casefold() == to_replace.casefold():
                        level = item.header["LVL_NUM"]
                        level_string = "lev" + str(level).replace(".", "p")
                        item.header["EXTNAME"] = level_string
                        # item.name = level_string
                        # hdul[item.name] = item
                        break

                # break
                #
                # item.data = np.empty(0)
                # item.name = to_replace

    def remove_unprocessed_frames(self, fits_path=None):
        # vprint("Blank Frame Ran")
        fits_path = fits_path or self.fits_path
        to_destroy = [
            "lev1p0",
            "t_int",
        ]

        with fits.open(
            fits_path, cache=False, ignore_missing_end=True, mode="update"
        ) as hdul:
            # hdul.verify('silentfix+ignore')  # Verify
            self.list_hdus(hdul)
            beginnings = [x.split("(")[0] for x in self.hdu_name_list]
            for name in to_destroy:
                sm = name.casefold()
                if sm in beginnings:
                    del hdul[name]

    def remove_unprocessed_frames2(self, fits_path=None):
        # vprint("Blank Frame Ran")
        fits_path = fits_path or self.fits_path
        to_destroy = ["lev1p0"]

        with fits.open(
            fits_path, cache=False, ignore_missing_end=True, mode="update"
        ) as hdul:
            # hdul.verify('silentfix+ignore')  # Verify
            self.list_hdus(hdul)
            for name in to_destroy:
                sm = name.casefold()
                for nn in self.hdu_name_list:
                    if sm in nn:
                        try:
                            del hdul[nn]
                        except KeyError as e:
                            print("KeyError", e)
                            # pass
        # try:
        #     frame = None
        #     if self.in_name is not None:
        #         if self.in_name in hdul:
        #             hdu = hdul[self.in_name]
        #         elif self.in_name =="lev1p0":
        #             in_name = "COMPRESSED_IMAGE"
        #             if in_name in hdul:
        #                 hdu = hdul[in_name]
        #         else:
        #             hdu = hdul[-1]
        #             self.params.png_frame_name = hdu.name
        #         frame = deepcopy(hdu.data)
        #     return frame
        # except OSError as e:
        #     print(e)
        #     return frame
        # except (UnboundLocalError, TypeError) as e:
        #     print("load single frame:: ", e)
        #     return None

    # Curves Save and Load
    def prep_save_outs(self):
        """Prepare the scalar_out_curve for writing"""
        if self.outer_min is None:
            return None
        self.scalar_out_curve = np.zeros(len(self.outer_min))
        if self.limb_radius_from_header:
            self.scalar_out_curve[0] = self.limb_radius_from_fit_shrunken
        if self.abs_min_scalar:
            self.scalar_out_curve[1] = self.abs_min_scalar
            self.scalar_out_curve[2] = self.abs_max_scalar
        if self.tri_filtered_inner_maximum is None:
            self.tri_filtered_outer_maximum = np.empty_like(self.outer_min)
            self.tri_filtered_inner_minimum = np.empty_like(self.outer_min)
            self.tri_filtered_inner_maximum = np.empty_like(self.outer_min)
            self.tri_filtered_outer_minimum = np.empty_like(self.outer_min)

        out_list = [
            self.outer_min,
            self.inner_min,
            self.inner_max,
            self.outer_max,
            self.scalar_out_curve,
        ]
        out_list.extend(
            [
                self.tri_filtered_outer_maximum,
                self.tri_filtered_inner_maximum,
                self.tri_filtered_inner_minimum,
                self.tri_filtered_outer_minimum,
                self.abs_max,
                self.abs_min,
            ]
        )
        # out_list.append([self.tri_filtered_absol_maximum, self.tii_filtered_absol_minimum])
        self.curve_descriptions = [
            "outer_min",
            "inner_min",
            "inner_max",
            "outer_max",
            ["scalar_out_curve", "limb_radius_from_fit_shrunken", "abs_min", "abs_max"],
            "tri_filtered_outer_maximum",
            "tri_filtered_inner_maximum",
            "tri_filtered_inner_minimum",
            "tri_filtered_outer_minimum",
            "smooth_abs_max",
            "smooth_abs_min",
        ]

        none_check = [item is not None for item in out_list]
        self.do_save = np.all(none_check)
        self.curve_out_array = np.asarray(out_list)
        return self.do_save

    def unpack_save_ins(self):
        """Prepare the scalar_out_curve for writing"""
        (
            self.outer_min,
            self.inner_min,
            self.inner_max,
            self.outer_max,
            self.scalar_in_curve,
            self.tri_filtered_outer_maximum,
            self.tri_filtered_inner_maximum,
            self.tri_filtered_inner_minimum,
            self.tri_filtered_outer_minimum,
            self.abs_max,
            self.abs_min,
        ) = np.loadtxt(self.params.curve_path())

        # self.limb_radius_from_file_shrunken = self.scalar_in_curve[0]
        self.limb_radius_from_file_shrunken = self.scalar_in_curve[0]
        self.abs_min_scalar = self.scalar_in_curve[1]
        self.abs_max_scalar = self.scalar_in_curve[2]

    def save_curves(self, banner=True, extra_line=False):  #
        """Save the curves so they don't have to be recalculated"""
        self.super_flush()
        if banner:
            if extra_line:
                vprint("\r *\n *    Saving Radial Curves...", end="")
            else:
                vprint("\r *        Saving Radial Curves...", end="")

        if self.prep_save_outs():
            curve_path = self.params.curve_path()
            descr_path = curve_path.replace("curve.txt", "curve_names.txt")
            makedirs(os.path.dirname(curve_path), exist_ok=True)

            with open(descr_path, mode="w") as fp:
                for desc, item in zip(self.curve_descriptions, self.curve_out_array):
                    len_item = str(len(item))
                    # len_desc = str(len(desc))
                    fp.write(str(desc) + " : len=" + len_item)
            np.savetxt(curve_path, self.curve_out_array)
            if banner:
                vprint("Success!")
        else:
            vprint("Skipping Save Curves!")

    def load_curves(self, force=None, verb=True):
        """Load the curves so they don't have to be recalculated"""
        lc = verb
        if os.path.exists(self.params.curve_path()):
            if self.abs_min_scalar is None or force:
                if lc:
                    print("\r *    Loading Radial Curves...", end="")
                try:
                    self.unpack_save_ins()
                    # if verb: self.super_flush("Success!\n")
                    if lc:
                        print("Success!", flush=True)
                    if False:
                        print("", flush=True)
                    self.curves_have_been_loaded = True
                except ValueError as e:
                    print("Failed to load Radial Curves: {}".format(e))
                    # raise e
        else:
            if True:
                print("No Curves to Load!")
                print("Please place the curves file at:")
                print(self.params.curve_path())

            # self.image_learn()
            # self.save_curves()

            # if hdul['primary'].data is None:
            #     hdul['primary'].data = hdul["lev1p0"].data + 0
            #     hdul[0].name = 'primary'
            #     hdul['primary'].header = hdul['primary'].header + hdul["lev1p0"].header
            #     del hdul["lev1p0"]
            #
            # # hdul.writeto(self.fits_path, output_verify="ignore", overwrite=True)
            #
            # for hdu in hdul:
            #
            #     try:
            #         del hdu.header['OSCNMEAN']
            #     except KeyError as e:
            #         pass
            #     try:
            #         del hdu.header['OSCNRMS' ]
            #     except KeyError as e:
            #         pass
            #

            # del hdul['primary'].header['OSCNMEAN']
            # del hdul['primary'].header['OSCNRMS']
            #
            # hdul[1].header['OSCNMEAN'] = 0.
            # hdul[1].header['OSCNRMS' ] = 0.
            #
            # hdul[0].verify('silentfix')
            # hdul[1].verify('silentfix')
            # a=1
            # hdul.verify('silentfix')
            #
            # print("mean  ", hdul[0].header['OSCNMEAN'])
            # print("rms   ", hdul[0].header['OSCNRMS' ])
            # print()
            # all_head_list0 = [('0 ', x, hdul[0].header[x]) for x in hdul[0].header]
            # all_head_list1 = [('1 ', x, hdul[1].header[x]) for x in hdul[1].header]
            # [print(x) for x in sorted(all_head_list0) if "OSCN" in x[1]]
            # [print(x) for x in sorted(all_head_list1) if "OSCN" in x[1]]
            #
            # # [print(x) for x in sorted(all_head_list) if "OSCN" in x]
            # a=1

            # print()
            # [print(x.name, '\t\t', x) for x in hdul]
            # a=1
            # print(self.list_hdus(hdul))
            # print()

            # data_frame = hdul[to_delete.pop(0)]
            # data_frame.name = 'PRIMARY'

            # for de in to_delete:
            #     del hdul[de]
            # hdul.update(self.fits_path)
            # [print(x.name, '\t\t', x) for x in hdul]

    def smallify_frame(self, frame):
        return frame
        mx = np.nanmax(frame)
        mn = np.nanmin(frame)
        normed = (frame - mn) / (mx - mn)

        scaled = normed * 2**16
        average = np.uint16(np.round(np.nanmean(scaled)))
        de_NANed = np.nan_to_num(scaled, nan=average)
        compressed = de_NANed.astype(np.uint16)

        return compressed

    def get_fits_info(self, hdul):
        """
        Extract FITS file information such as wavelength, observation time, and more.

        Args:
            hdul (astropy.io.fits.HDUList): The FITS HDU list.

        Returns:
            tuple: A tuple containing wavelength, observation time, center coordinates,
            integration time, and limb radius.
        """
        wave, t_rec, center, int_time, found_limb_radius = None, None, None, None, None

        for hdu in hdul:
            try:
                if "lev" in hdu.name:
                    last_hdul_frame = hdul[hdu.name]
                else:
                    last_hdul_frame = hdul[-1]  # Fall back to using an index
                last_hdul_frame.header["DRMS_ID"]
                self.header = last_hdul_frame.header
                wave = last_hdul_frame.header["WAVELNTH"]
                t_rec = last_hdul_frame.header["T_OBS"]
                center = [
                    last_hdul_frame.header["X0_MP"],
                    last_hdul_frame.header["Y0_MP"],
                ]
                int_time = last_hdul_frame.header["EXPTIME"]
                found_limb_radius = last_hdul_frame.header["R_SUN"]
                self.params.bunit = last_hdul_frame.header["BUNIT"]
                while found_limb_radius > last_hdul_frame.header["NAXIS1"]:
                    found_limb_radius /= 4.0
                break
            except KeyError:
                # print("NNNNNN")
                continue

        self.params.limb_radius_from_header = found_limb_radius
        self.params.header = self.header
        # print(center)
        return wave, t_rec, center, int_time, found_limb_radius

    def load_this_fits_frame(self, fits_path=None, in_name=None, quiet=False):
        """
        Load a FITS file from disk and extract information.

        Args:
            fits_path (str, optional): The path to the FITS file.
            in_name (str, optional): The name of the desired HDU.
            quiet (bool, optional): Whether to suppress output.

        Returns:
            tuple: A tuple containing the loaded frame, wavelength, observation time,
            center coordinates, integration time, and frame name.
        """
        try:
            fits_path = fits_path.replace(".jpg", ".fits")
            with fits.open(
                fits_path,
                cache=False,
                ignore_missing_end=True,
                ignore_missing_simple=True,
                memmap=False,
            ) as hdul:
                self.hdu_name_list = self.list_hdus(hdul)
                self.in_name = self.set_in_frame_name(
                    in_name=in_name, fits_path=fits_path, hdul=hdul
                )

                frame, self.header = self.open_fits_hdul(
                    hdul=hdul, quiet=quiet, frame_name=self.in_name
                )

                wave, t_rec, center, int_time, self.limb_radius_from_header = (
                    self.get_fits_info(hdul)
                )
                hdul.verify()
            return frame, wave, t_rec, center, int_time, self.in_name
        except (FileNotFoundError, FileExistsError) as e:
            print("\n", e)
            print("HDU's found: ", self.hdu_name_list, "\n")
            pass
        except (OSError, RuntimeError) as e:
            pass
            print("\n", e)
            # print("Unable to load Frame!")
        except (TypeError, Exception) as e:
            print("\n", e)
        self.skipped += 1
        return None, None, None, None, None, None

    def set_in_frame_name(self, in_name=None, fits_path=None, hdul=None):
        """
        Determine the frame name based on different input types.

        Args:
            in_name (str or int or list, optional): The input name or index.
            fits_path (str, optional): The path to the FITS file.
            hdul (astropy.io.fits.HDUList, optional): The FITS HDU list.

        Returns:
            str: The determined frame name.
        """
        if isinstance(in_name, str):
            if in_name.isdigit():
                in_name = self.hdu_name_list[int(in_name)]
            self.in_name = self.frame_name = in_name
            return self.in_name


        if hdul is None:
            with fits.open(fits_path, cache=False, ignore_missing_end=True) as hdul:
                self.in_name = self.find_correct_in_name(hdul, name=in_name)
        else:
            self.in_name = self.find_correct_in_name(hdul, name=in_name)

        return self.in_name

    def outer(self, name, do=False):
        """
        Remove parentheses from a frame name.

        Args:
            name (str): The frame name.
            do (bool): Whether to remove parentheses.

        Returns:
            str: The frame name with or without parentheses.
        """
        if self.do_split or do:
            return name.split("(")[0]
        return name

    def get_frame_names(self, requested_output_name, do_split=False):
        """
        Get various frame names based on the requested output name.

        Args:
            requested_output_name (str): The requested output frame name.
            do_split (bool, optional): Whether to remove parentheses from frame names.

        Returns:
            tuple: A tuple containing the first frame name, second frame name, penultimate frame name,
            last frame name, previous frame name, and a list of all frame names.
        """
        self.do_split = do_split

        # Get the first frame name
        first_name = self.outer(self.hdu_name_list[0])

        # Get the second frame name or use the first if there is only one frame
        if len(self.hdu_name_list) > 1:
            second_name = self.outer(self.hdu_name_list[1])
        else:
            second_name = first_name

        # Get the penultimate frame name
        penultimate_name = self.outer(self.determine_penultimate_frame_name())

        # Get the last frame name
        last_name = self.outer(self.hdu_name_list[-1])

        try:
            # Create a list of cleaned, casefolded names for comparison
            sh_all_names = [
                self.outer(x.casefold(), do=True)
                for x in self.hdu_name_list
                if isinstance(x, str)
            ]

            # Find the previous frame name based on the requested output name
            prev_name = self.outer(
                self.hdu_name_list[sh_all_names.index(requested_output_name) - 1]
            )

        except ValueError:
            # If the requested output name is not found, use the penultimate frame name
            prev_name = self.outer(penultimate_name)

        # Create a list of cleaned, casefolded names for all frame names
        all_names = [
            self.outer(x.casefold()) for x in self.hdu_name_list if isinstance(x, str)
        ]

        return (
            first_name,
            second_name,
            penultimate_name,
            last_name,
            prev_name,
            all_names,
        )

    def find_correct_in_name(self, hdul, name):
        """
        Determine the correct input frame name for redoing operations.

        Args:
            hdul (astropy.io.fits.HDUList): The FITS HDU list.
            name (str or None): The requested input name.

        Returns:
            str: The determined input frame name.
        """
        repo = self.reprocess_mode()
        reprocess_mode = self.params.reprocess_mode(repo)

        # List all the various Names
        self.hdu_name_list = self.list_hdus(hdul)
        requested_input_name = self.determine_in_frame_name(hdul, name)
        requested_output_name = self.outer(self.determine_out_frame_name())

        first_name, second_name, penultimate_name, last_name, prev_name, all_names = (
            self.get_frame_names(requested_output_name)
        )

        (
            sh_first_name,
            sh_second_name,
            sh_penultimate_name,
            sh_last_name,
            sh_prev_name,
            sh_all_names,
        ) = self.get_frame_names(requested_output_name, True)

        # Do logic
        filter_already_applied = requested_output_name.casefold() in sh_all_names
        if filter_already_applied:
            if reprocess_mode in ["skip", False]:
                # Skip it
                self.in_name = None
                raise FileExistsError
            elif reprocess_mode in ["redo", None, True]:
                # Go to the previous out_array and remake
                self.in_name = prev_name
            elif reprocess_mode == "reset":
                # Go to the first out_array and remake
                self.in_name = first_name
            elif reprocess_mode == "double":
                # Repeat the filter a second time
                self.in_name = requested_output_name
            elif reprocess_mode == "add":
                # Repeat the filter a second time
                self.in_name = requested_output_name
                self.out_name = self.out_name + "_redo"
            else:
                raise NotImplementedError
        else:
            self.in_name = requested_input_name or self.in_name
        hdul.verify("silentfix+ignore")
        return self.in_name

    def determine_in_frame_name(self, hdul, name, quiet=True):
        """
        Parse an in_array variable to determine the frame it's referring to.

        Args:
            hdul (astropy.io.fits.HDUList): The FITS HDU list.
            name (str or int or list): The input name or index.

        Returns:
            str: The determined frame name.
        """
        self.frame_name = None
        self.hdu_name_list = self.list_hdus(hdul)

        if name is None:
            return None

        if isinstance(name, str):
            self.frame_name = name

        elif isinstance(name, int):
            offset = 0 if name < len(self.hdu_name_list) else 1
            self.frame_name = self.hdu_name_list[name - offset]

        elif isinstance(name, list):
            self.frame_name = self.pick_from_list(name, quiet)

        return self.frame_name

    def pick_from_list(self, name, quiet=True):
        """This function selects the appropriate frame to use from a list of names"""

        for try_two in [False, True]:
            for input_name in name:
                input_name = input_name.casefold()
                short_input_name = input_name.split("(")[0]

                to_check = short_input_name if try_two else input_name

                for full_name in self.hdu_name_list:
                    short_name = full_name.split("(")[0]

                    if to_check in short_name:  # or name in lowercase_hdu_names:
                        self.frame_name = full_name
                        if not quiet:
                            print("\r +    Using frame {}".format(self.frame_name))
                        break
                if self.frame_name is not None:
                    return self.frame_name
        return None

    @staticmethod
    def clean_time_string(time_string, targetZone=None, out_fmt=None):
        # Make the name strings
        import pytz

        # Ingest the original time in UTC
        original = datetime.strptime(time_string.split(".")[0], "%Y-%m-%dT%H:%M:%S")
        tz_UTC = pytz.timezone("UTC")
        original = original.replace(tzinfo=tz_UTC)

        if targetZone is not None:
            tz_diff = pytz.timezone(targetZone)
            cleaned = original.astimezone(tz_diff)
        else:
            cleaned = original

        default_out_fmt = "%I:%M%p %Z,  %m-%d-%y"
        out_fmt = out_fmt or default_out_fmt
        out_str = cleaned.strftime(out_fmt)

        return out_str

    def open_fits_hdul_old(self, hdul, quiet=True, fail=False, frame_name=None):
        """Load a fits file from disk"""
        # self.rename_initial_frames(hdul)
        self.frame_name = frame_name or self.frame_name
        self.hdu_name_list = self.list_hdus(hdul)
        hdu_name_list_trimmed = [x.split("(")[0] for x in self.hdu_name_list]
        if self.frame_name is None:
            print("asdf")
        # if self.frame_name is None:
        #     return None, None
        # if self.frame_name.casefold() == 'primary':
        #     self.frame_name = self.hdu_name_list[1]
        name = self.pick_from_list(self.params.master_frame_list_newest)

        try:
            # Try it as Written
            field_hdu = hdul[self.frame_name]
        except KeyError as e:
            try:
                # Try shortening the in_array name
                field_hdu = hdul[self.frame_name.split("(")[0]]
            except KeyError as e2:
                try:
                    # Try shortening the file frame names

                    loc = [x == self.frame_name for x in hdu_name_list_trimmed]
                    idx = np.where(loc)[0][0]

                    field_hdu = hdul[idx]

                except KeyError:
                    try:
                        field_hdu = hdul[self.in_name]
                    except KeyError as e:
                        if not quiet:
                            print("Oh No! Can't Find {}".format(self.frame_name))
                        if fail:
                            raise e
                    field_hdu = hdul[name]
                # found = False
                # for name in self.params.master_frame_list_newest:
                #     for item in self.hdu_name_list:
                #         if name in item:
                #             field_hdu = hdul[item]
                #             self.frame_name = item
                #             if not quiet:
                #                 print("Using {} instead".format(item))
                #             found = True
                #             break
                #     if found:
                #         break
                # if not found:
                #     raise e2
                #
        data = None
        header = None
        field_hdu = None or field_hdu
        if field_hdu.data is None:
            if name == "primary":
                try:
                    field_hdu = hdul[0]
                    if field_hdu.data is None:
                        field_hdu = hdul[1]
                    if field_hdu.data is None:
                        field_hdu = hdul[1]
                except KeyError as e:
                    field_hdu = hdul["lev1p0"]
        try:
            data = field_hdu.data + 0
            header = hdul[1].header
        except TypeError:
            vprint("Processor: 1224 !Failed to Load Frame!")
        except IndexError:
            data = field_hdu.data + 0
            header = hdul[0].header

        # if Processor.print_once:
        #     print("\r +    Loading Frame: {}".format(self.frame_name))
        #     self.print_once = False
        return data, header

    # def get_field_hdu(self, hdul, frame_name=None):
    #     # Gather Names
    #     self.frame_name = frame_name or self.frame_name
    #     self.trimmed_name = self.frame_name.split('(')[0]
    #     self.hdu_name_list = self.list_hdus(hdul)
    #     self.hdu_name_list_trimmed = [x.split('(')[0] for x in self.hdu_name_list]

    #     # # Determine which frames are available
    #     # frames = self.hdu_name_list
    #     # frames_trimmed = self.hdu_name_list_trimmed

    #         # [name for name in self.params.master_frame_list_oldest
    #         #               if name in self.hdu_name_list_trimmed]

    #     # Try to get frame
    #     field_hdu = None
    #     if self.frame_name in self.hdu_name_list:
    #         # Try it as Written
    #         field_hdu = hdul[self.frame_name]

    #         loc = [x == self.frame_name for x in self.hdu_name_list]
    #         idx = np.where(loc)[0][0]

    #     elif self.trimmed_name in self.hdu_name_list:

    #         field_hdu = hdul[self.trimmed_name]

    #         loc = [x == self.trimmed_name for x in self.hdu_name_list_trimmed]
    #         idx = np.where(loc)[0][0]

    #     elif self.trimmed_name in self.hdu_name_list_trimmed:
    #         loc = [x == self.trimmed_name for x in self.hdu_name_list_trimmed]
    #         idx = np.where(loc)[0][0]
    #         field_hdu = hdul[idx]
    #         self.frame_name = self.trimmed_name
    #     else:
    #         raise FileNotFoundError

    #     # Make sure not to fumble the first frame
    #     early_names = ["primary", '', 'lev1p0']
    #     idxx = 0
    #     if field_hdu is not None and self.frame_name in early_names:
    #         try:
    #             while field_hdu.data is None:
    #                 field_hdu = hdul[idxx]
    #                 idxx += 1
    #         except ValueError:
    #             print(field_hdu.__getattribute__('data'))
    #             pass
    #     return field_hdu, self.frame_name, idx

    # def open_fits_hdul(self, hdul, quiet=True, fail=False, frame_name=None):
    #     """Load a fits file from disk"""

    #     # self.rename_initial_frames(hdul)

    #     field_hdu, frame_name, loc = self.get_field_hdu(hdul, frame_name)

    #     data = field_hdu.data
    #     header = field_hdu.header
    #     return data, header

    # def list_hdus(self, hdul):
    #     if hdul is not None:
    #         hdul.verify('silentfix+ignore')  # Verify
    #         # self.rename_initial_frames(hdul)  # This might not work
    #         self.hdu_name_list = [frame.name.casefold() for frame in hdul]
    #         hdul.verify('silentfix+ignore')  # Verify
    #     return self.hdu_name_list

    def _get_cleaned_frame_name(self):
        # Extract the frame name and remove parentheses if present
        return self.frame_name.split("(")[0] if self.frame_name else None

    def get_field_hdu(self, hdul, frame_name=None):
        """
        Retrieve a specific Header Data Unit (HDU) from a FITS file.

        Args:
            hdul (astropy.io.fits.HDUList): The FITS HDU list.
            frame_name (str, optional): The name of the desired HDU.

        Returns:
            astropy.io.fits.PrimaryHDU or astropy.io.fits.ImageHDU: The selected HDU.
            str: The selected frame name.
            int: The index of the selected HDU in the HDU list.

        Raises:
            FileNotFoundError: If the desired HDU is not found in the file.
        """
        if frame_name is not None:
            self.frame_name = frame_name
        cleaned_frame_name = self._get_cleaned_frame_name()

        special_names = ["primary", "compressed_image", "lev1p5"]
        found_hdu = None
        found_idx = -1

        for idx, hdu in enumerate(hdul):
            hduname = hdu.name.casefold()
            cleaned_hdu_name = hduname.split("(")[0]

            # Check if current HDU matches the requested frame name
            if hduname == self.frame_name.casefold() or cleaned_hdu_name == cleaned_frame_name.casefold():
                if hdu.data is None:
                    if hduname in special_names or cleaned_hdu_name in special_names:
                        self.frame_name = "compressed_image"
                        continue
                    raise FileNotFoundError(
                        "The specified HDU '{}' was found but has no data.".format(
                            self.frame_name
                        )
                    )
                return hdu, hdu.name, idx

            # Track the last HDU matching special names
            if hduname in special_names or cleaned_hdu_name in special_names:
                found_hdu = hdu
                found_idx = idx

        # Return the last matching special name HDU if found
        if found_hdu is not None:
            return found_hdu, found_hdu.name, found_idx

        # Raise an error if no appropriate HDU was found
        raise FileNotFoundError(
            "The specified HDU '{}' was not found.".format(self.frame_name)
        )


    def open_fits_hdul(self, hdul, quiet=True, frame_name=None, peek=False):
        """
        Load data and header from a specific HDU in a FITS file.

        Args:
            hdul (astropy.io.fits.HDUList): The FITS HDU list.
            quiet (bool, optional): Whether to suppress output.
            frame_name (str, optional): The name of the desired HDU.

        Returns:
            numpy.ndarray: The data from the selected HDU.
            astropy.io.fits.Header: The header from the selected HDU.

        Raises:
            FileNotFoundError: If the desired HDU is not found in the file.
        """
        # import pdb; pdb.set_trace()

        field_hdu, frame_name, _ = self.get_field_hdu(hdul, frame_name)
        data = field_hdu.data
        header = field_hdu.header

        if peek:
            plt.imshow(data, cmap="viridis")
            plt.title(frame_name)
            plt.show(block=True)
        return data, header

    def list_hdus(self, hdul):
        """
        Get a list of HDU names (casefolded) from a FITS HDU list.

        Args:
            hdul (astropy.io.fits.HDUList): The FITS HDU list.

        Returns:
            List[str]: A list of HDU names (in lowercase).
        """
        hdul.verify("silentfix+ignore")  # Verify the FITS HDU list
        hdu_name_list = [hdu.name.casefold() for hdu in hdul]
        self.hdu_name_list = hdu_name_list
        return hdu_name_list

    def determine_penultimate_frame_name(self):
        get = -2 if len(self.hdu_name_list) > 1 else -1
        return self.hdu_name_list[get]

    def determine_first_frame_name(self, hdul=None):
        self.list_hdus(hdul)
        return self.hdu_name_list[0]

    def determine_last_frame_name(self, hdul=None):
        self.list_hdus(hdul)
        return self.hdu_name_list[-1]

    def determine_requested_in_frame_name(self):
        # Determine the called-for in_array out_array NAME
        if type(self.in_name) is str:
            # if self.in_name in self.hdu_name_list:
            input_frame_name = self.in_name.casefold()
        elif type(self.in_name) not in [list, type(None)]:
            input_frame_name = self.hdu_name_list[self.in_name].casefold()
        else:
            input_frame_name = self.in_name
        return input_frame_name

    def determine_out_frame_name(self):
        # Determine the called-for output out_array NAME
        if type(self.out_name) is str:
            output_frame_name = self.out_name.casefold()
        elif self.out_name is not None:
            output_frame_name = self.hdu_name_list[self.out_name].casefold()
        else:
            output_frame_name = "None"
        return output_frame_name

    def determine_first_hIndex(self, hdul):
        """Find out which hInd has the data"""
        hInd = 0
        for hInd in range(10):
            try:
                a = hdul[hInd].header["WAVELNTH"]
                a = hdul[hInd].data
                break
            except Exception as e:
                pass
        return hInd

    ## UTIL
    def get(self):
        """Return just the modified_image frome"""
        return self.params.modified_image

    def get_orig(self):
        """Return just the raw_image frome"""
        return self.params.raw_image

    def super_flush(self, txt=None, end=None, many=5):
        """Flush the stdout many times"""
        if txt:
            print(txt, flush=True, end=end)
        for ii in range(many):
            sys.stdout.flush()
            sys.stderr.flush()

    def printout_hdul(self, hdul):
        print("\n\n**Examining Hdul**")

        for h_num in range(len(hdul)):
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n")
            print("\n  HDUL #", h_num)
            for h_info in hdul.fileinfo(h_num):
                print(
                    "    ",
                    h_num,
                    " : ",
                    h_info,
                    "\t : \t",
                    hdul.fileinfo(h_num)[h_info],
                )

            to_find = ["bound method", "built-in method", "method-wrapper"]
            HDU = hdul[h_num]

            print("\n  Hdul Fields")
            self.print_without(HDU, to_find)

            for ff in to_find:
                self.print_with(HDU, ff)

    def view_raw(self, fig=None, ax=None):
        if fig is None and ax is None:
            fig, ax = plt.subplots(num="Input Image")
        ax.set_title("Preview of Start Frame: {}".format(self.params.hdu_name))
        minmin = np.min(self.params.raw_image)
        img = np.sqrt(np.asarray(self.params.raw_image - minmin, dtype=np.float32))
        ax.imshow(img, cmap=self.params.cmap)

    def print_without(self, HDU, not_wanted=None):
        print("      ** " + "Remainder without these:  " + str(not_wanted) + " **")
        ban_list = [str(it) for it in not_wanted]

        found_list = []
        for found_field_name in dir(HDU):
            found_field_value = getattr(HDU, found_field_name)
            found_field_value_string = str(found_field_value)
            skip = False
            for bb in ban_list:
                if bb in found_field_value_string:
                    skip = True
            if skip:
                continue

            found_list.append([found_field_name, found_field_value])

            # Print these
            out_string = "      {}".format("Misc") + ":: " + found_field_name
            out_2 = "\t  :  \t" + found_field_value_string
            print("{0: <35}".format(out_string.replace("\n", " ")), out_2)
        print("\n\n")
        return found_list

    def print_with(self, HDU, wanted=None):
        print("    ** " + str(wanted) + " **")
        found_list = []
        for found_field_name in dir(HDU):
            found_field_value = getattr(HDU, found_field_name)
            found_field_value_string = str(found_field_value)

            if wanted in str(found_field_value_string):
                found_list.append([found_field_name, found_field_value])
                out_string = "      {}".format(wanted) + ":: " + found_field_name
                out_2 = "\t  :  \t" + found_field_value_string
                print("{0: <35}".format(out_string), out_2)
        print("\n\n")

        return found_list

    @staticmethod
    def write_video_in_directory(
        directory=None,
        file_name=None,
        fps=10,
        pop=None,
        folder_name=None,
        desc=None,
        key_string="keyframe",
        fullpath=None,
        destroy=False,
        shortcut=False,
        orig=False,
    ):
        """Make a video out of whatever directory it's pointed at"""
        video_avi = None
        file_name = file_name or "default_videoname.avi"
        video_path = None
        try:
            if fullpath is not None:
                folder = os.path.dirname(fullpath)
                good_paths = [
                    join(folder, f)
                    for f in listdir(folder)
                    if ("png" in f and not os.path.isdir(join(folder, f)))
                ]
                video_path = fullpath.replace(".png", ".avi")
            else:
                radial_directory = directory
                # makedirs(radial_directory, exist_ok=True)
                video_path = radial_directory + "\\" + file_name
                good_paths = [
                    radial_directory + "\\" + f
                    for f in listdir(radial_directory)
                    if "png" in f
                ]

            if orig:
                video_path = os.path.normpath(
                    os.path.join(
                        directory, r"..\\..\\..\\video\\orig_{}".format(file_name)
                    )
                )

                if desc is None:
                    desc = " *    Writing Video {}".format(basename(directory))
            if video_path is None:
                if pop:
                    filename = os.path.basename(video_path)
                    directory = os.path.dirname(video_path)
                    up_dir_1 = os.path.dirname(directory)
                    up_dir_2 = os.path.dirname(up_dir_1)
                    up_dir_3 = os.path.dirname(up_dir_2)

                    if pop is True:
                        up_dir = up_dir_1
                    if pop == 2:
                        up_dir = up_dir_2
                    if pop == 3:
                        up_dir = up_dir_3

                    video_path = os.path.join(up_dir, "video", filename)

            # Initialize the Machine
            if len(good_paths):
                good_paths.sort()
                first_path = good_paths[0]
                height, width, _ = cv2.imread(first_path).shape
                video_avi = cv2.VideoWriter(video_path, 0, fps, (width, height))

                # Write the Frames
                for img_path in tqdm(good_paths, desc=desc, unit="frames"):
                    video_avi.write(cv2.imread(img_path))
                    if destroy:
                        os.remove(img_path)
                    # for img_path in good_paths:
            else:
                print("VideoProcessor:: There are no images yet. Make them first.")
                1 + 1
        except FileNotFoundError as e:
            print("Processor.py:", e)
        finally:
            # Shut it all down
            cv2.destroyAllWindows()
            if video_avi is not None:
                video_avi.release()
            # if shortcut:
            #     import winshell
            #     self.params.basename()
        # print(" ^    Successfully {} from {} images! ({} skipped)".format(self.finished_verb, ii, self.skipped))

    def orig_smasher(self, orig):
        return np.log10(orig) / 2

    def touchup_TUNE(self, img):
        print("TOUCHUP TOOOOOOOOON")
        img *= 10.0
        np.power(img, 1 / 3, out=img)
        img /= 3.5

        # img += 0.1

        # img[img > 1.] = np.power(img[img > 1.], 1/2)

        # img *= 1.5
        # img -= 0.75

        # img[img < 0.] = 0.
        # img[img == 0.] = np.nan
        img[~np.isfinite(img)] = np.nan
        return img

    @staticmethod
    def despike(arr, n1=2.5, n2=40, block=25):
        def rolling_window(data, block):
            shape = data.shape[:-1] + (data.shape[-1] - block + 1, block)
            strides = data.strides + (data.strides[-1],)
            return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

        # Condition the Input
        data = arr.copy()
        data[data == -1] = np.NaN
        offset = np.nanmin(data)
        data -= offset
        roll = rolling_window(data, block)
        roll = np.ma.masked_invalid(roll)
        std = n1 * roll.std(axis=1)
        mean = roll.mean(axis=1)
        # Use the last value to fill-up.
        std = np.r_[std, np.tile(std[-1], block - 1)]
        mean = np.r_[mean, np.tile(mean[-1], block - 1)]
        mask = np.abs(data - mean.filled(fill_value=np.NaN)) > std.filled(
            fill_value=np.NaN
        )
        data[mask] = np.NaN
        # Pass two: recompute the mean and std without the flagged values from pass
        # one now removing the flagged data.
        roll = rolling_window(data, block)
        roll = np.ma.masked_invalid(roll)
        std = n2 * roll.std(axis=1)
        mean = roll.mean(axis=1)
        # Use the last value to fill-up.
        std = np.r_[std, np.tile(std[-1], block - 1)]
        mean = np.r_[mean, np.tile(mean[-1], block - 1)]
        mask = np.abs(arr - mean.filled(fill_value=np.NaN)) > std.filled(
            fill_value=np.NaN
        )
        arr[mask] = mean[mask]
        return arr + offset

    def maximizePlot(self):
        try:
            mng = plt.get_current_fig_manager()
            backend = plt.get_backend()
            if backend == "TkAgg":
                try:
                    mng.window.state("zoomed")
                except:
                    mng.resize(*mng.window.maxsize())
            elif backend == "wxAgg":
                mng.frame.Maximize(True)
            elif backend[:2].upper() == "QT":
                mng.window.showMaximized()
            else:
                return False
            return True
        except:
            return False

    @staticmethod
    def norm_formula(image, the_min, the_max):
        """Standard Normalization Formula"""
        image_flat = image.flatten()
        diff = np.subtract(the_max, the_min)
        np.subtract(image_flat, the_min, out=image_flat)
        np.divide(image_flat, diff, out=image_flat)
        image = image_flat.reshape(image.shape)
        return image

    def vignette(self, frame=None):
        """Truncate the in_object above a certain radis"""
        # if self.vignette_mask is None:
        return frame

        if self.radius is None:
            self.init_radius_array()

        if frame is not None:
            # frame = frame.astype(np.float16)
            mask = np.nan if "float" in str(frame.dtype) else 0
            if len(frame.shape) > 2:
                frame[:, self.vignette_mask] = mask
            else:
                frame[self.vignette_mask] = mask
            return frame

        else:
            self.params.modified_image.astype(np.float16)[self.vignette_mask] = np.nan
            self.params.raw_image.astype(np.float16)[self.vignette_mask] = np.nan

            if self.params.rhe_image is not None:
                self.params.rhe_image[self.vignette_mask] = np.nan
            if self.params.rbg_image is not None:
                self.params.rbg_image[self.vignette_mask] = 1
            return None

    ## Static Methods ##
    def n2r_fp(self, n):
        """Convert index to solar radius"""
        if not self.limb_radius_from_fit_shrunken:
            self.find_limb_radius()
        if n is None:
            n = 0
        r = n / self.limb_radius_from_fit_shrunken_forpoints
        return r

    def r2n_fp(self, r):
        """Convert index to solar radius"""
        if not self.limb_radius_from_fit_shrunken:
            self.find_limb_radius()
        n = r * self.limb_radius_from_fit_shrunken_forpoints
        return n

    def n2r(self, n):
        """Convert index to solar radius"""
        if not self.limb_radius_from_fit_shrunken:
            self.find_limb_radius()
        if n is None:
            n = 0
        r = n / self.limb_radius_from_fit_shrunken
        return r

    def r2n(self, r):
        """Convert index to solar radius"""
        if not self.limb_radius_from_fit_shrunken:
            self.find_limb_radius()
        n = r * self.limb_radius_from_fit_shrunken
        return n

    @staticmethod
    def normalize(image, high=99.99, low=5.0):
        """Normalize the Array"""
        lowP, highP = np.nanpercentile(image, [low, high])

        # if low is None:
        #     lowP = 0
        # else:
        #     lowP = np.nanpercentile(use_image, low)
        # highP = np.nanpercentile(use_image, high)
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                out = (image - lowP) / (highP - lowP)
            except RuntimeWarning as e:
                out = image
        return out

    @staticmethod
    def fill_end(use):
        iii = -1
        val = use[iii]
        while np.isnan(val):
            iii -= 1
            val = use[iii]
        use[iii:] = val
        return use

    @staticmethod
    def fill_start(use):
        iii = 0
        val = use[iii]
        while np.isnan(val):
            iii += 1
            val = use[iii]
        use[:iii] = val
        return use

        # fail_count = 0
        # img_paths = self.params.local_imgs_paths()
        # for ii, img_path in enumerate(tqdm(img_paths, desc="  ")):
        #     try:
        #         self.modify_one_img(img_path, self.do_fits_function)
        #     except Exception as e:
        #         print(e)
        #         fail_count += 1
        #     self.ii = ii
        # print("    Success! {} Files Processed\n".format(self.ii+1))

    # def modify_one_img(self, img_path, function):
    #     in_object = function(img_path, self.in_field).get()
    #     # save_frame_to_fits_file(img_path, in_object, get_field=self.out_name)
    #     return in_object


# def process_fits(self, params=None):
#     if params is not None:
#         self.params = params
#         load_fits_paths(self.params)
#
#     print(self.name+"...", flush=True)
#     self.modify_fits_series()
#     print("    Success! {} Files Filtered\n".format(self.ii+1))

# def modify_fits_series(self):
#     """Processes the fits series"""
#     for ii, img_path in enumerate(tqdm(self.params.local_fits_paths(), desc="  ")):
#         try:
#             in_object = self.modify_fits(img_path, self.in_field)
#         except Exception as e:
#             print(e)
#         self.ii = ii

# def modify_fits(self, img_path, function, in_field=None, out_name=None):
#     if in_field: self.in_field = in_field
#     if out_name: self.out_name = out_name
#     in_object = function(img_path, self.in_field).get()
#     save_frame_to_fits_file(img_path, in_object, get_field=self.out_name)
#     return in_object

# def modify_one_fits(self, img_path, function, in_field=None, out_name=None):
#     raise NotImplementedError()


# def build_paths(self, wave):
#     self.local_wave_directory = join(self.params.imgs_top_directory(), wave)
#     self.image_folder = join(self.local_wave_directory, 'png')
#     self.movie_folder = abspath(join(self.params.imgs_top_directory(), "movies\\"))
#     self.video_name_stem = join(self.movie_folder, '{}_{}_movie{}'.format(wave, strftime('%m%d_%H%M'), '{}'))
#     # print(self.video_name_stem)
#     makedirs(self.movie_folder, exist_ok=True)

# images = [img for img in listdir(self.image_folder) if img.endswith(".png")] # and self.check_valid_png(img)]
