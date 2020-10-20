"""
sunback.py
A program that downloads the most current images of the sun from the SDO satellite,
then sets each of the images to the desktop background in series.

Handles the primary functions
"""

# Imports
from time import localtime, timezone, strftime, sleep, time, struct_time
from urllib.request import urlretrieve
from os import getcwd, makedirs, rename, remove, listdir, startfile
from os.path import normpath, abspath, join, dirname, exists
from calendar import timegm
import astropy.units as u

start = time()
from sunpy.net import Fido, attrs as a
import sunpy.map
from sunpy.io import read_file_header, write_file
from moviepy.editor import AudioFileClip, VideoFileClip
import cv2
from pippi import tune
from functools import partial
from threading import Thread, Barrier
from copy import copy
bbb = Barrier(3, timeout=100)

print("Import took {:0.2f} seconds".format(time() - start))
from parfive import Downloader
import numba
# from numba import jit
from tqdm import tqdm
from warnings import warn
from platform import system
import sys
import numpy as np
import matplotlib as mpl

try:
    mpl.use('qt5agg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

from playsound import playsound as ps
from pippi.oscs import Osc
from pippi import dsp, fx

this_system = system()

if this_system == "Windows":
    # Windows Imports
    import sunpy.visualization.colormaps

elif this_system == "Linux":
    # Linux Imports
    import sunpy.visualization.colormaps

elif this_system == "Darwin":
    # Mac Imports
    pass

else:
    raise OSError("Operating System Not Supported")

# Main

debugg = False



def tr():
    import pdb;
    pdb.set_trace()


class Parameters:
    """
    A container class for the run parameters of the program
    """
    seconds = 1
    minutes = 60 * seconds
    hours = 60 * minutes

    def __init__(self):
        """Sets all the attributes to None"""
        # Initialize Variables
        self.background_update_delay_seconds = None
        self.time_multiplier_for_long_display = None
        self.local_directory = None
        self.use_wavelengths = None
        self._resolution = 4096
        self.web_image_frame = None
        self.web_image_location = None
        self.web_paths = None
        self.file_ending = None
        self.run_time_offset = None
        self.time_file = None
        self.index_file = None
        self.debug_mode = False

        self.start_time = time()
        self.is_first_run = True
        self._do_HMI = True
        self._mode = 'all'
        self._do_mirror = False

        # Movie Defaults
        self._download_images = True
        self._overwrite_pngs = False
        self._make_compressed = False
        self._remove_old_images = False
        self._sonify_images = True
        self._sonify_limit = True
        self._do_171 = False
        self._do_304 = False
        self._something_changed = False
        self._allow_muxing = True

        self._stop_after_one = False

        self._time_period = None
        self._range_in_days = 5
        self._cadence = 10 * u.minute
        self._frames_per_second = 30
        self._bpm = 70

        self.set_default_values()

    def download_images(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._download_images = boolean
        if self._download_images:
            self.something_changed(True)
        return self._download_images

    def something_changed(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._something_changed = boolean
        return self._something_changed

    def overwrite_pngs(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._overwrite_pngs = boolean
        if self._overwrite_pngs:
            self.something_changed(True)
        return self._overwrite_pngs

    def make_compressed(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._make_compressed = boolean
        return self._make_compressed

    def remove_old_images(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._remove_old_images = boolean
        if self._remove_old_images:
            if self.something_changed():
                return True
        return False

    def sonify_images(self, boolean=None, mux=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._sonify_images = boolean
        if mux is not None:
            self.allow_muxing(mux)
        return self._sonify_images

    def allow_muxing(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._allow_muxing = boolean
        return self._allow_muxing

    def do_mirror(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._do_mirror = boolean
        return self._do_mirror

    def sonify_limit(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._sonify_limit = boolean
        return self._sonify_limit

    def do_171(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._do_171 = boolean
            if self._do_171:
                self.stop_after_one(True)
        return self._do_171

    def do_304(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._do_304 = boolean
            if self._do_304:
                self.stop_after_one(True)
        return self._do_304

    def stop_after_one(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._stop_after_one = boolean
        return self._stop_after_one

    def range(self, days=None, hours=None):
        if days is not None or hours is not None:
            total_days = 0
            if days is not None:
                total_days += days
            if hours is not None:
                total_days += hours / 24
            self._range_in_days = total_days
        return self._range_in_days

    def cadence(self, cad=None):
        if cad is not None:
            self._cadence = cad * u.minute
        return self._cadence

    def time_period(self, period=None):
        if period is not None:
            self._time_period = period
        return self._time_period

    def frames_per_second(self, rate=None):
        if rate is not None:
            self._frames_per_second = rate
        return self._frames_per_second

    def bpm(self, bpm=None):
        if bpm is not None:
            self._bpm = bpm
        return self._bpm

    def check_real_number(self, number):
        assert type(number) in [float, int]
        assert number > 0

    def set_default_values(self):
        """Sets the Defaults for all the Parameters"""
        # SunbackMovie Parameters

        # Sunback Still Parameters
        #  Set Delay Time for Background Rotation
        self.set_delay_seconds(30 * self.seconds)
        self.set_time_multiplier(3)

        # Set File Paths
        self.set_local_directory()
        self.time_file = join(self.local_directory, 'time.txt')
        self.index_file = join(self.local_directory, 'index.txt')

        # Set Wavelengths
        self.set_wavelengths(['0171', '0193', '0211', '0304', '0131', '0335', '0094', 'HMIBC', 'HMIIF'])

        # Set Resolution
        self.set_download_resolution(2048)

        # Set Web Location
        self.set_web_image_frame("https://sdo.gsfc.nasa.gov/assets/img/latest/latest_{}_{}")

        # Add extra images
        new_web_path_1 = "https://sdo.gsfc.nasa.gov/assets/img/latest/f_211_193_171pfss_{}.jpg".format(self.resolution)
        self.append_to_web_paths(new_web_path_1, 'PFSS')

        # Select File Ending
        self.set_file_ending("{}_Now.png")

        return 0

    # Methods that Set Parameters
    def set_delay_seconds(self, delay):
        self.check_real_number(delay)
        self.background_update_delay_seconds = delay
        return 0

    def set_time_multiplier(self, multiplier):
        self.check_real_number(multiplier)
        self.time_multiplier_for_long_display = multiplier
        return 0

    def set_local_directory(self, path=None):
        if path is not None:
            self.local_directory = path
        else:
            self.local_directory = self.discover_best_default_directory()

        makedirs(self.local_directory, exist_ok=True)

    def set_wavelengths(self, waves):
        # [self.check_real_number(int(num)) for num in waves]
        self.use_wavelengths = waves
        self.use_wavelengths.sort()
        if self.has_all_necessary_data():
            self.make_web_paths()
        return 0

    def set_download_resolution(self, resolution):
        self.check_real_number(resolution)
        self._resolution = min([170, 256, 512, 1024, 2048, 3072, 4096], key=lambda x: np.abs(x - resolution))
        if self.has_all_necessary_data():
            self.make_web_paths()

    def resolution(self, resolution=None):
        if resolution is not None:
            self.check_real_number(resolution)
            self._resolution = min([170, 256, 512, 1024, 2048, 3072, 4096], key=lambda x: np.abs(x - resolution))
        return self._resolution

    def set_web_image_frame(self, path):
        self.web_image_frame = path
        if self.has_all_necessary_data():
            self.make_web_paths()

    def set_file_ending(self, string):
        self.file_ending = string

    # Methods that create something

    def make_web_paths(self):
        self.web_image_location = self.web_image_frame.format(self.resolution, "{}.jpg")
        self.web_paths = [self.web_image_location.format(wave) for wave in self.use_wavelengths]

    def append_to_web_paths(self, path, wave=' '):
        self.web_paths.append(path)
        self.use_wavelengths.append(wave)

    # Methods that return information or do something
    def has_all_necessary_data(self):
        if self.web_image_frame is not None:
            if self.use_wavelengths is not None:
                if self.resolution is not None:
                    return True
        return False

    def get_local_path(self, wave):
        return normpath(join(self.local_directory, self.file_ending.format(wave)))

    @staticmethod
    def discover_best_default_directory():
        """Determine where to store the images"""

        subdirectory_name = "sunback_images"
        if __file__ in globals():
            ddd = dirname(abspath(__file__))
        else:
            ddd = abspath(getcwd())

        while "dropbox".casefold() in ddd.casefold():
            ddd = abspath(join(ddd, ".."))

        directory = join(ddd, subdirectory_name)

        # print("Image Location: {}".format(directory))
        # while not access(directory, W_OK):
        #     directory = directory.rsplit(sep)[0]
        #
        # print(directory)
        return directory

    def determine_delay(self):
        """ Determine how long to wait """

        delay = self.background_update_delay_seconds + 0
        # import pdb; pdb.set_trace()
        # if 'temp' in wave:
        #     delay *= self.time_multiplier_for_long_display

        self.run_time_offset = time() - self.start_time
        delay -= self.run_time_offset
        delay = max(delay, 0)
        return delay

    def wait_if_required(self, delay):
        """ Wait if Required """

        if self.is_first_run:
            self.is_first_run = False
        elif delay <= 0:
            pass
        else:
            # print("Took {:0.1f} seconds. ".format(self.run_time_offset), end='')
            print("Waiting for {:0.0f} seconds ({} total)".format(delay, self.background_update_delay_seconds),
                  flush=True, end='')

            fps = 3
            for ii in (range(int(fps * delay))):
                sleep(1 / fps)
                print('.', end='', flush=True)
                # self.check_for_skip()
            print('Done')

    def sleep_until_delay_elapsed(self):
        """ Make sure that the loop takes the right amount of time """
        self.wait_if_required(self.determine_delay())

    def is_debug(self, debug=None):
        if debug is not None:
            self.debug_mode = debug
        return self.debug_mode

    def do_HMI(self, do=None):
        if do is not None:
            self._do_HMI = do
        return self._do_HMI

    def mode(self, mode=None):
        if mode is not None:
            self._mode = mode
        return self._mode

class SunbackMovie:
    """
    The Primary Class that Does Everything

    Parameters
    ----------
    parameters : Parameters (optional)
        a class specifying run options
    """

    def __init__(self, parameters=None):
        """Initialize a new parameter object or use the provided one"""
        self.fits_analysis_done = False
        self.indexNow = 0
        if parameters:
            self.params = parameters
        else:
            self.params = Parameters()

        self.last_time = 0
        self.this_time = 1
        self.new_images = False
        self.fido_result = None
        self.fido_num = None
        self.renew_mask = True
        self.mask_num = [1, 2]
        self.wavelengths = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700']
        self.waveNum = len(self.wavelengths)
        self.this_name = None
        self.resume = True
        self.sonify_complete = False
        self.proper_bin=None

        if not self.params.is_debug():
            sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')

    # # Main Command Structure  ##############################################
    def start(self):
        """Select whether to run or to debug"""
        self.print_header()

        if self.params.is_debug():
            self.debug_mode()
        else:
            self.run_mode()

    def debug_mode(self):
        """Run the program in a way that will break"""
        while True:
            self.execute()

    def run_mode(self):
        """Run the program in a way that won't break"""

        fail_count = 0
        fail_max = 10

        while True:
            try:
                self.execute()
            except (KeyboardInterrupt, SystemExit):
                print("\n\nOk, I'll Stop. Doot!\n")
                break
            except Exception as error:
                fail_count += 1
                if fail_count < fail_max:
                    print("I failed, but I'm ignoring it. Count: {}/{}\n\n".format(fail_count, fail_max))
                    self.need = True
                    continue
                else:
                    print("Too Many Failures, I Quit!")
                    sys.exit(1)

    def print_header(self):
        print("\nSunback: Live SDO Background Updater \nWritten by Chris R. Gilly")
        print("Check out my website: http://gilly.space\n")
        # print("Delay: {} Seconds".format(self.params.background_update_delay_seconds))
        print("Coronagraph Mode: {} \n".format(self.params.mode()))

        if self.params.is_debug():
            print("DEBUG MODE\n")

    def execute(self):
        if self.resume:
            self.resume_last_index()
        while self.indexNow < self.waveNum:
            self.indexNow += 1
            self.main_loop(self.indexNow - 1)
            self.save_current_index()
        self.indexNow = 0
        self.save_current_index()

    def resume_last_index(self):
        try:
            with open(self.params.index_file, 'r') as fp:
                self.indexNow = int(fp.readline())
        except Exception as e:
            self.indexNow = 0
            warn("Failed to resume from last wavelength.")

    def save_current_index(self):
        with open(self.params.index_file, 'w') as fp:
            fp.write(str(self.indexNow))

    # # Main Functions  ######################################################

    def main_loop(self, ii):
        """The Main Loop"""

        # Initialize Everything
        if self.init_or_skip(ii):
            return

        # Download the new fits data
        self.space_to_fits()

        # Remove the old fits files
        self.remove_all_old_files()

        # Analyze the Dataset as a whole
        if False:
            self.fits_analyze_whole_set()

        # Generate the png images
        self.fits_to_pngs()

        # Generate the Movie
        self.pngs_to_movie()

        # Generate the Audio
        self.fits_to_audio()

        # Add Sound to the Movie
        self.movie_to_audio_movie()

        self.soni.thread_lock()
        print("Wavelength Complete: {}, took {:0.2f} minutes\n\n".format(self.this_name, (time() - self.beginTime) / 60))

        if self.params.stop_after_one():
            sys.exit()

    def init_or_skip(self, ii):
        """Initializes the instance variables to this index if the flags allow it, else skips"""
        if self.params.do_171():
            ii = 2

        if self.params.do_304():
            ii = 5

        self.this_name = self.wavelengths[ii]

        if self.skip():
            return True

        self.beginTime = time()
        self.save_path = join(self.params.local_directory, self.this_name)
        makedirs(self.save_path, exist_ok=True)
        self.local_path = self.params.local_directory
        self.image_folder = self.save_path
        self.movie_folder = self.save_path
        self.video_name_stem = join(self.movie_folder, '{}_movie{}'.format(self.this_name, '{}'))

        name = "{}_{}".format(self.this_name, "max")
        self.soni = Sonifier(self.params, self.save_path, name, self.video_name_stem, frames_per_second=self.params.frames_per_second())

        print("\nMovie: {}".format(self.this_name))
        return False

    def space_to_fits(self):
        """ Find the Most Recent Images """
        # Define Time Range
        self.get_time_range()

        if self.params.download_images():
            print(">Acquiring Science Images from {} to {}...".format(self.earlyString, self.nowString), flush=True)

            # Search for records from the internet
            self.fido_result = Fido.search(a.Time(self.early, self.now), a.Instrument('aia'), a.Wavelength(int(self.this_name) * u.angstrom),
                                           a.vso.Sample(self.params.cadence()), a.Level(1.5))

            # See what we already have
            files = listdir(self.save_path)
            already_downloaded = []
            for filename in files:
                if filename.endswith(".fits") and "norm" not in filename:
                    already_downloaded.append(int(self.time_from_filename(filename)[0]))

            # Define what we still need
            to_get = np.empty(self.fido_result.file_num)
            to_get.fill(True)
            for ii, file_to_get in enumerate(self.fido_result.get_response(0)):
                tm = int(file_to_get.time.start)
                if tm in already_downloaded:
                    to_get[ii] = False
            getNum =int(np.sum(to_get)) - 1
            need = np.nonzero(to_get)[0]

            # Analyze Results
            self.import_fido_information()
            print("  Search Found {} Images {}...".format(self.fido_num, self.extra_string), flush=True)

            if getNum > 1:
                self.new_images = True
                print("    Downloading {} New:".format(getNum), flush=True)

                #Make groups
                start = need[0]
                end=start
                box2=[]
                box1 = []

                for ii in np.arange(len(need)-1):
                    end = need[ii]+1
                    if need[ii] == need[ii+1] - 1:
                        continue
                    box2.append(self.fido_result[0, start:end])
                    box1.append((start,end))
                    start = need[ii+1]
                if end - start > 1:
                    box2.append(self.fido_result[0, start:end])
                    box1.append((start,end))

                for st in box2:
                    Fido.fetch(st, path=self.save_path, downloader=Downloader(progress=True, file_progress=False), overwrite=True)
                # print("Short took {:0.3f} seconds.".format(time()-startT))
            else:
                self.new_images = False
                print("   Success: All Images Already Downloaded", flush=True)


    def get_time_range(self):
        """Define Time Range, on the hour"""
        # Get the Start Time
        current_time = time() + timezone
        start_list = list(localtime(current_time - (self.params.range() + 2 / 24) * 60 * 60 * 24))

        # Truncate the minutes and seconds
        # start_list[2] -= 0 # Days

        start_list[4] = 0  # Minutes
        start_list[5] = 0  # Seconds
        start_struct = struct_time(start_list)

        # Make Output Products
        self.early = strftime('%Y/%m/%d %H:%M', start_struct)
        self.earlyLong = int(strftime('%Y%m%d%H%M%S', start_struct))
        self.earlyString = self.parse_time_string_to_local(str(self.earlyLong), 2)

        # Get the Current Time
        now_list = list(localtime(current_time - 2 * 60 * 60))

        # Truncate the minutes and seconds
        now_list[4] = 0  # Minutes
        now_list[5] = 0  # Seconds
        now_struct = struct_time(now_list)

        # Make Output Products
        self.now = strftime('%Y/%m/%d %H:%M', now_struct)
        self.nowLong = int(strftime('%Y%m%d%H%M%S', now_struct))
        self.nowString = self.parse_time_string_to_local(str(self.nowLong), 2)

    def fits_analyze_whole_set(self):
        """Check several fits files to determine fit curves"""
        print("Analyzing Dataset...", flush=True, end='')
        max_analyze = 5
        minBox = []
        maxBox = []

        self.save_path = join(self.params.local_directory, self.this_name)

        ii = 0
        for filename in listdir(self.save_path):
            if filename.endswith(".fits"):
                ii += 1
                image_path = join(self.save_path, filename)
                fname = filename[3:18]
                fname = fname.replace("_", "")
                time_string = self.parse_time_string_to_local(fname, 2)

                # Load the File
                originalData, single_image_data = self.load_fits_series((self.this_name, image_path, time_string))

                self.radial_analyze(originalData, False)
                minBox.append(self.fakeMin)
                maxBox.append(self.fakeMax)

                print('.', end='')
                if ii >= max_analyze:
                    self.fakeMin = np.mean(np.asarray(minBox), axis=0)
                    self.fakeMax = np.mean(np.asarray(maxBox), axis=0)
                    self.fits_analysis_done = True
                    print("Done!")
                    break

    def fits_to_pngs(self):
        """Re-save all the Fits images into pngs and normed fits files"""
        self.apply_func_to_directory(self.do_image_work, doAll=False, desc=">Processing Images", unit="images")

    def fits_to_audio(self):
        """Analyzes the fits files to sonify them"""
        if self.params.sonify_images() and not self.sonify_complete:
            self.apply_func_to_directory(self.do_sonifying_work, doAll=True, desc=">Sonifying Images", unit="images", limit=self.params.sonify_limit())
        self.soni.generate_track(self.soni.wav_path)
            # self.soni.play()

    def pngs_to_movie(self):
        """Combines all png files into an avi movie"""
        try:
            videoclip_full = VideoFileClip(self.video_name_stem.format("_raw.avi"))
            invalid_movie=False
        except:
            invalid_movie=True

        if self.new_images or invalid_movie:
            # logger = open(join(self.params.local_directory, 'log.txt'), 'w+')
            try:
                images = [img for img in listdir(self.image_folder) if img.endswith(".png") and self.check_valid_png(img)]
                if len(images) > 0:
                    frame = cv2.imread(join(self.image_folder, images[0]))
                    height, width, layers = frame.shape
                    video_avi = cv2.VideoWriter(self.video_name_stem.format("_raw.avi"), 0, self.params.frames_per_second(), (width, height))

                    for image in tqdm(images, desc=">Writing Movie", unit="frame"):
                        # Delete it if it is too old
                            im = cv2.imread(join(self.image_folder, image))
                            # import pdb; pdb.set_trace()
                            video_avi.write(im)


                    cv2.destroyAllWindows()
                    video_avi.release()

                else:
                    print("No png Images Found")
            except FileNotFoundError:
                print("Images Not Found")

    def movie_to_audio_movie(self):
        """Multiplexes the generated wav and avi files into a single movie"""
        if self.params.allow_muxing() and (self.new_images or self.params.sonify_images()):
            print(">Muxing Main Movie...")
            videoclip_full = VideoFileClip(self.video_name_stem.format("_raw.avi"))
            videoclip_full_muxed = videoclip_full.set_audio(AudioFileClip(self.soni.wav_path))
            from proglog import TqdmProgressBarLogger

            hq_sonFunc = partial(videoclip_full_muxed.write_videofile, self.video_name_stem.format("_HQ.mp4"), codec='libx264', bitrate='400M',
                                                 logger=TqdmProgressBarLogger(print_messages=True))
            t1 = Thread(target=hq_sonFunc)
            t1.start()
            t1.join()


            if self.params.make_compressed():
                clip = videoclip_full_muxed
                lq_sonFunc = partial(clip.write_videofile, self.video_name_stem.format("_LQ.mp4"), codec='libx264', bitrate='50M')
                wq_sonFunc = partial(clip.write_videofile, self.video_name_stem.format(".webm"), codec='libvpx')
                t2 = Thread(target=lq_sonFunc)
                t3 = Thread(target=wq_sonFunc)
                t2.start()
                t2.join()
                t3.start()
                t3.join()



            print("  Successfully Muxed")
            # # Play the Movie
            # startfile(self.video_name_stem.format("_HQ.mp4"))

    # # Support Functions  ###################################################

    def skip(self):

        # Skip Logic
        # if '4500' in self.this_name:
        #     return
        # if self.this_name in ['0094'] and self.params.is_first_run:
        #     # print("Skip for Now\n")
        #     return 1
        # #
        # # if self.params.is_debug():
        # if self.this_name in ['0193']:
        #     sys.exit()
        # #
        # # # if int(self.this_name) < 1000:
        # # #     return

        # if '304' not in self.this_name:
        #     return 1
        return 0

    def import_fido_information(self):
        self.fido_num = self.fido_result.file_num

        time_start = self.fido_result.get_response(0)[0].time.start
        time_end = self.fido_result.get_response(0)[-1].time.start

        self.startTime = self.parse_time_string_to_local(str(int(time_start)), 2)
        self.endTime = self.parse_time_string_to_local(str(int(time_end)), 2)

        self.name = self.fido_result.get_response(0)[0].wave.wavemin
        while len(self.name) < 4:
            self.name = '0' + self.name

        if self.params.is_debug():
            self.extra_string = "from {} to {}".format(self.startTime, self.endTime)
        else:
            self.extra_string = ''

    def define_single_image(self, filename):
        time_code, time_string = self.time_from_filename(filename)
        image_path = join(self.save_path, filename)
        single_image_data = (self.this_name, image_path, time_string, time_code, filename)
        return single_image_data

    def remove_all_old_files(self):
        files = listdir(self.save_path)
        file_idx = 0
        for filename in files:
            if filename.endswith(".fits") and "norm" not in filename:
                if self.remove_old_files(self.define_single_image(filename)):
                    file_idx += 1
                    continue
        if file_idx > 0:
            if self.params.remove_old_images():
                print("Deleted {} old images".format(file_idx))
            # else:
            #     print("Excluding {} old images".format(file_idx))

    def remove_old_files(self, single_image_data):
        filename = single_image_data[4]
        thisTime = int(self.time_from_filename(filename)[0])
        if thisTime < self.earlyLong:
            if self.params.remove_old_images():
                self.deleteFiles(filename)
            return 1
        return 0

    def this_frame_is_bad(self, image_array, single_image_data):
        filename = single_image_data[4]
        total_counts = np.nansum(image_array)
        if total_counts < 0:
            self.deleteFiles(filename)
            return 1
        return 0

    def apply_func_to_directory(self, func, doAll=False, desc="Work Done", unit="it", limit=False):
        work_list = []
        files = listdir(self.save_path)
        file_idx = -1
        for filename in files:
            if filename.endswith(".fits") and "norm" not in filename:
                # Define the image
                single_image_data = self.define_single_image(filename)
                pngPath = join(self.save_path, filename[:-4] + 'png')

                # Delete it if it is too old
                if self.remove_old_files(single_image_data):
                    continue

                file_idx += 1
                if doAll or not exists(pngPath) or self.params.overwrite_pngs():
                    if not limit or self.soni.frame_on_any_beat(file_idx):
                        work_list.append([single_image_data, file_idx])
                    else:
                        work_list.append(None)

        self.nRem = len(work_list)

        if self.nRem > 0:
            with tqdm(total=self.nRem, desc=desc, unit=unit) as pbar:
                for image in work_list:
                    if image:
                        func(image)
                    pbar.update()
            # # pbar.close()
            # from pymp import Parallel
            # with tqdm(total=self.nRem, desc=desc, unit=unit) as pbar:
            #     with Parallel(self.nRem) as p:
            #         for i in p.range(self.nRem):
            #             image = work_list[i]
            #             if image:
            #                 func(image)
            #             pbar.update()
            #     # pbar.close()
            # from joblib import Parallel, delayed

            # results = Parallel(n_jobs=-1, verbose=verbosity_level, backend="threading")(
            #     map(delayed(myfun), arg_instances))

            # import threading as mp
            # from multiprocessing.pool import ThreadPool
            # pool = ThreadPool()
            # pool.map(func, work_list)

    def do_image_work(self, single_image_data_ID):
        single_image_data, file_idx = single_image_data_ID
        # Load the File, destroying it if it fails
        fail, raw_image = self.load_fits_series(single_image_data)
        if fail:
            return 1

        # # Remove bad frames
        if self.this_frame_is_bad(raw_image, single_image_data):
            return 1

        # Modify the data
        processed_image_stats = self.image_modify(raw_image)

        # Sonify the data
        if False: #self.params.sonify_images():
            self.do_sonifying_work(single_image_data_ID, processed_image_stats, raw_image)
            self.sonify_complete=True if not self.params.download_images() else False

        # Plot and save the Data
        self.plot_and_save(processed_image_stats, single_image_data, raw_image)
        self.new_images = True
        return 0

    def do_sonifying_work(self, single_image_data_ID, proc_image_stats=None, raw_image=None):
        single_image_data, file_idx = single_image_data_ID

        if raw_image is None:
            # Load the File, destroying it if it fails

            fail1, raw_image = self.load_fits_series(single_image_data)

            # # Remove bad frames
            if fail1 or self.this_frame_is_bad(raw_image, single_image_data):
                return 1

        if proc_image_stats is None:
            single_image_data_proc = list(single_image_data)
            # print(single_image_data_proc)
            # import pdb; pdb.set_trace()
            single_image_data_proc[1] = single_image_data_proc[1][:-5] + "_norm.fits"
            fail2, proc_image_stats = self.load_fits_series(single_image_data_proc)

        # Sonify the data
        self.soni.sonify_frame(proc_image_stats, raw_image, file_idx)

        return 0

    def load_fits_series(self, image_data):
        # Load the Fits File from disk
        full_name, save_path, time_string, time_code, filename = image_data
        try:
            # Parse Inputs
            my_map = sunpy.map.Map(save_path)
        except (TypeError, OSError) as e:
            remove(save_path)
            return 1, 1

        data = my_map.data
        return 0, data

    def deleteFiles(self, filename):
        fitsPath = join(self.save_path, filename[:-5] + '.fits')
        pngPath = join(self.save_path, filename[:-5] + '.png')
        fitsPath2 = join(self.save_path, filename[:-5] + '_norm.fits')

        try:
            remove(fitsPath)
        except:
            pass
        try:
            remove(fitsPath2)
        except:
            pass
        try:
            remove(pngPath)
        except:
            pass

    def time_from_filename(self, filename):
        fname = filename[3:18]
        time_code = fname.replace("_", "")
        time_string = self.parse_time_string_to_local(time_code, 2)
        return time_code, time_string

    def check_valid_png(self, img):
        image_is_new=(int(self.time_from_filename(img)[0])) < self.earlyLong
        return not image_is_new

    def image_modify(self, data):
        data = data + 0
        self.radial_analyze(data, False)
        data = self.absqrt(data)
        data = self.coronagraph(data)
        data = self.vignette(data)
        data = self.append_stats(data)
        return data

    def append_stats(self, data):
        from scipy.signal import savgol_filter
        rank = 1
        window1 = 31
        window2 = 41
        mode = 'mirror'
        btma = self.binBtm[::self.extra_rez]
        mina = self.binMin[::self.extra_rez]
        mida = self.binMid[::self.extra_rez]
        maxa = self.binMax[::self.extra_rez]
        topa = self.binTop[::self.extra_rez]

        btma = savgol_filter(btma, window1, rank, mode=mode)
        mina = savgol_filter(mina, window1, rank, mode=mode)
        mida = savgol_filter(mida, window1, rank, mode=mode)
        maxa = savgol_filter(maxa, window1, rank, mode=mode)
        topa = savgol_filter(topa, window1, rank, mode=mode)

        btma = savgol_filter(btma, window2, rank, mode=mode)
        mina = savgol_filter(mina, window2, rank, mode=mode)
        mida = savgol_filter(mida, window2, rank, mode=mode)
        maxa = savgol_filter(maxa, window2, rank, mode=mode)
        topa = savgol_filter(topa, window2, rank, mode=mode)

        stacked = np.vstack(
            (data, btma, mina, mida, maxa, topa))
        return stacked

    def plot_and_save(self, data, image_data, original_data=None, ii=None):
        full_name, save_path, time_string, time_code, filename = image_data
        name = self.clean_name_string(full_name)

        for processed in [True]:

            if not self.params.is_debug():
                if not processed:
                    continue
            if not processed:
                if original_data is None:
                    continue

            # Save the Fits File
            header = read_file_header(save_path)[0]
            if "BLANK" in header.keys():
                del header["BLANK"]
            path = save_path[:-5] + '_norm.fits'
            write_file(path, np.asarray(data, dtype=np.float32), header, overwrite=True)

            data, _ = self.soni.remove_stats(data)

            # Create the Figure
            fig, ax = plt.subplots()
            self.blankAxis(ax)

            # inches = 10
            # fig.set_size_inches((inches, inches))
            #
            # pixels = data.shape[0]
            # dpi = pixels / inches
            # cocaine
            siX = 10
            siY = 10
            piX = 1080
            piY = 1080
            dpX = piX / siX
            dpY = piY / siY
            dpi = np.max((dpX, dpY))
            fig.set_size_inches((siX, siY))

            if 'hmi' in name.casefold():
                inst = ""
                plt.imshow(data, origin='upper', interpolation=None)
                # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                plt.tight_layout(pad=5.5)
                height = 1.05

            else:
                inst = '  AIA'
                plt.imshow(data if processed else self.normalize(original_data), cmap='sdoaia{}'.format(name),
                           origin='lower', interpolation=None,
                           vmin=self.vmin_plot, vmax=self.vmax_plot)
                plt.tight_layout(pad=0)
                height = 0.95

            # Annotate with Text
            buffer = '' if len(name) == 3 else '  '
            buffer2 = '    ' if len(name) == 2 else ''

            title = "{}    {} {}, {}{}".format(buffer2, inst, name, time_string, buffer)
            title2 = "{} {}, {}".format(inst, name, time_string)
            ax.annotate(title, (0.125, height + 0.02), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')
            # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
            # the_time = strftime("%I:%M%p").lower()
            # if the_time[0] == '0':
            #     the_time = the_time[1:]
            # ax.annotate(the_time, (0.125, height), xycoords='axes fraction', fontsize='large',
            #             color='w', horizontalalignment='center')

            # Format the Plot and Save
            self.blankAxis(ax)
            middle = '' if processed else "_orig"
            new_path = save_path[:-5] + middle + ".png"

            if ii is not None and self.nRem > 0:
                remString = "{} / {} , {:0.1f}%".format(ii, self.nRem, 100 * ii / self.nRem)
            else:
                remString = ""

            try:
                plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi, compression=2, filter=None)
                # print("\tSaved {} Image {}, {} of {}   ".format('Processed' if processed else "Unprocessed", time_string, remString, self.this_name), end="\r")
            except PermissionError:
                new_path = save_path[:-5] + "_b.png"
                plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
                print("Success")
            except Exception as e:
                print("Failed...using Cached")
                if self.params.is_debug():
                    raise e
            plt.close(fig)

        return new_path

    def update_background(self, local_path, test=False):
        """
        Update the System Background

        Parameters
        ----------
        local_path : str
            The local save location of the image
        """
        print("Updating Background...", end='', flush=True)
        assert isinstance(local_path, str)
        local_path = normpath(local_path)

        this_system = platform.system()

        try:
            if this_system == "Windows":
                import ctypes
                ctypes.windll.user32.SystemParametersInfoW(20, 0, local_path, 0)
                # ctypes.windll.user32.SystemParametersInfoW(19, 0, 'Fill', 0)

            elif this_system == "Darwin":
                from appscript import app, mactypes
                try:
                    app('Finder').desktop_picture.set(mactypes.File(local_path))
                except Exception as e:
                    if test:
                        pass
                    else:
                        raise e

            elif this_system == "Linux":
                import os
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-options 'scaled'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background primary-color 'black'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-uri {}".format(local_path))
            else:
                raise OSError("Operating System Not Supported")
            print("Success")
        except Exception as e:
            print("Failed")
            raise e

        if self.params.is_debug():
            self.plot_stats()

        return 0

    # Level 4

    @staticmethod
    def list_files1(directory, extension):
        from os import listdir
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    def get_paths(self, this_result):
        self.name = this_result.get_response(0)[0].wave.wavemin
        while len(self.name) < 4:
            self.name = '0' + self.name
        file_name = '{}_Now.fits'.format(self.name)
        save_path = join(self.params.local_directory, file_name)
        return self.name, save_path

    @staticmethod
    def parse_time_string_to_local(downloaded_files, which=0):
        if which == 0:
            time_string = downloaded_files[0][-25:-10]
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = int(time_string[9:11])
            minute = time_string[11:13]
        else:
            time_string = downloaded_files
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = time_string[8:10]
            minute = time_string[10:12]

        struct_time = (int(year), int(month), int(day), int(hour_raw), int(minute), 0, 0, 0, -1)

        new_time_string = strftime("%I:%M%p %m/%d/%Y", localtime(timegm(struct_time))).lower()
        if new_time_string[0] == '0':
            new_time_string = new_time_string[1:]

        # print(year, month, day, hour, minute)
        # new_time_string = "{}:{}{} {}/{}/{} ".format(hour, minute, suffix, month, day, year)

        return new_time_string

    @staticmethod
    def clean_name_string(full_name):
        # Make the name strings
        name = full_name + ''
        while name[0] == '0':
            name = name[1:]
        return name

    @staticmethod
    def blankAxis(ax):
        ax.patch.set_alpha(0)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='none', which='both',
                       top=False, bottom=False, left=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Data Manipulations

    @staticmethod
    def reject_outliers(data):
        # # Reject Outliers
        # a = data.flatten()
        # remove_num = 20
        # ind = argpartition(a, -remove_num)[-remove_num:]
        # a[ind] = nanmean(a)*4
        # data = a.reshape(data.shape)

        data[data > 10] = np.nan

        return data

    @staticmethod
    def absqrt(data):
        return np.sqrt(np.abs(data))

    @staticmethod
    def normalize(data):
        high = 98
        low = 15

        lowP = np.nanpercentile(data, low)
        highP = np.nanpercentile(data, high)
        return (data - lowP) / (highP - lowP)

    def vignette(self, data):

        mask = self.radius > (self.noise_radii)
        data[mask] = np.nan
        return data

    def vignette2(self, data):

        mask = np.isclose(self.radius, self.tRadius, atol=2)
        data[mask] = 1

        mask = np.isclose(self.radius, self.noise_radii, atol=2)
        data[mask] = 1
        return data

    def coronagraph(self, data):
        original = sys.stderr
        sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')

        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        dat_corona = (self.dat_flat - self.fakeMin[radius_bin]) / \
                     (self.fakeMax[radius_bin] - self.fakeMin[radius_bin])

        sys.stderr = original

        # Deal with too hot things
        self.vmax = 1
        self.vmax_plot = 1.9

        hot = dat_corona > self.vmax
        dat_corona[hot] = dat_corona[hot] ** (1 / 2)

        # Deal with too cold things
        self.vmin = 0.06
        self.vmin_plot = -0.03

        cold = dat_corona < self.vmin
        dat_corona[cold] = -((np.abs(dat_corona[cold] - self.vmin) + 1) ** (1 / 2) - 1) + self.vmin

        self.dat_coronagraph = dat_corona
        dat_corona_square = dat_corona.reshape(data.shape)

        if self.renew_mask or self.params.mode() == 'r':
            self.corona_mask = self.get_mask(data)
            self.renew_mask = False

        data = self.normalize(data)

        data[self.corona_mask] = dat_corona_square[self.corona_mask]

        # inds = np.argsort(self.rad_flat)
        # rad_sorted = self.rad_flat[inds]
        # dat_sort = dat_corona[inds]
        #
        # plt.figure()
        # # plt.yscale('log')
        # plt.scatter(rad_sorted[::30], dat_sort[::30], c='k')
        # plt.show()

        return data

    def get_mask(self, dat_out):

        corona_mask = np.full_like(dat_out, False, dtype=bool)
        rezz = corona_mask.shape[0]
        half = int(rezz / 2)

        mode = self.params.mode()

        if type(mode) in [float, int]:
            mask_num = mode
        elif 'y' in mode:
            mask_num = 1
        elif 'n' in mode:
            mask_num = 2
        else:
            if 'r' in mode:
                if len(mode) < 2:
                    mode += 'a'

            if 'a' in mode:
                top = 8
                btm = 1
            elif 'h' in mode:
                top = 6
                btm = 3
            elif 'd' in mode:
                top = 8
                btm = 7
            elif 'w' in mode:
                top = 2
                btm = 1
            else:
                print('Unrecognized Mode')
                top = 8
                btm = 1

            ii = 0
            while True:
                mask_num = np.random.randint(btm, top + 1)
                if mask_num not in self.mask_num:
                    self.mask_num.append(mask_num)
                    break
                ii += 1
                if ii > 10:
                    self.mask_num = []

        if mask_num == 1:
            corona_mask[:, :] = True

        if mask_num == 2:
            corona_mask[:, :] = False

        if mask_num == 3:
            corona_mask[half:, :] = True

        if mask_num == 4:
            corona_mask[:half, :] = True

        if mask_num == 5:
            corona_mask[:, half:] = True

        if mask_num == 6:
            corona_mask[:, :half] = True

        if mask_num == 7:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True

        if mask_num == 8:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True
            corona_mask = np.invert(corona_mask)

        return corona_mask

    # Basic Analysis

    def radial_analyze(self, data, plotStats=False):

        self.offset = np.abs(np.min(data))
        data += self.offset
        self.make_radius(data)
        self.sort_radially(data)

        stats = self.better_bin_stats(self.rad_sorted, self.dat_sorted, self.rez, self.offset)
        self.binBtm, self.binMin, self.binMax, self.binMid, self.binTop = stats

        if not self.fits_analysis_done:
            self.fit_curves()

        if plotStats:
            self.plot_stats()

    def make_radius(self, data):

        self.rez = data.shape[0]
        centerPt = self.rez / 2
        xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
        xc, yc = xx - centerPt, yy - centerPt

        self.extra_rez = 2

        self.sRadius = 400 * self.extra_rez
        self.tRadius = self.sRadius * 1.28
        self.radius = np.sqrt(xc * xc + yc * yc) * self.extra_rez
        self.rez *= self.extra_rez

    def sort_radially(self, data):
        # Create arrays sorted by radius
        self.rad_flat = self.radius.flatten()
        self.dat_flat = data.flatten()
        inds = np.argsort(self.rad_flat)
        self.rad_sorted = self.rad_flat[inds]
        self.dat_sorted = self.dat_flat[inds]



    @staticmethod
    # @numba.jit(nopython=True, parallel=True)
    def better_bin_stats(rad_sorted, dat_sorted, rez, offset):
        proper_bin = np.asarray(np.floor(rad_sorted), dtype=np.int32)
        binBtm = np.empty(rez);
        binBtm.fill(np.nan)
        binMin = np.empty(rez);
        binMin.fill(np.nan)
        binMax = np.empty(rez);
        binMax.fill(np.nan)
        binMid = np.empty(rez);
        binMid.fill(np.nan)
        binTop = np.empty(rez);
        binTop.fill(np.nan)

        bin_list = [np.float64(x) for x in range(0)]
        last = 0
        for ii in np.arange(len(proper_bin)):
            binInd = proper_bin[ii]
            if binInd != last:
                bin_array = np.asarray(bin_list)
                finite = bin_array[np.isfinite(bin_array)]
                data_in_bin = finite[np.nonzero(finite - offset)]
                if len(data_in_bin) > 0:
                    out = np.percentile(data_in_bin, [0.001, 2, 50, 95, 99.999])
                    binBtm[last], binMin[last], binMid[last], binMax[last], binTop[last] = out
                bin_list = []
            bin_list.append(dat_sorted[ii])
            last = binInd
        return binBtm, binMin, binMax, binMid, binTop

        # self.radBins = [[] for x in np.arange(self.rez)]
        # for ii, binI in enumerate(self.proper_bin):
        #     self.radBins[binI].append(self.dat_sorted[ii])
        #
        # # Find the statistics by bin
        # for bin_count, bin_list in enumerate(self.radBins):
        #     self.bin_the_slice(bin_count, bin_list)

        # i = 0
        # bin_count = 0
        # not_edge = self.proper_bin[:-1] == self.proper_bin[1:]
        # theRez = len(self.proper_bin)
        # myList = []
        # while i < theRez:
        #     if i < theRez - 1 and not_edge[i]:
        #         i += 1
        #         myList.append(self.dat_sorted[i])
        #         continue
        #     self.bin_the_slice(bin_count, np.asarray(myList))
        #     bin_count += 1
        #     i += 1
        # i = 0
        # i_prev = 0
        # bin_count = 0
        # not_edge = self.proper_bin[:-1] == self.proper_bin[1:]
        # theRez = len(self.proper_bin)
        # myList = []
        # while i < theRez:
        #     if i < theRez - 1 and not_edge[i]:
        #         i += 1
        #         continue
        #     bin_arr = self.dat_sorted[i_prev:i + 1]
        #     self.bin_the_slice(bin_count, bin_arr)
        #     bin_count += 1
        #     i += 1

        # top = int(np.ceil(np.max(self.rad_sorted)))
        # print("Top : {}".format(top))
        # bin_edges = np.arange(top)  # or whatever
        # self.binMin = np.empty(bin_edges.size - 1)
        # self.binMax = np.empty(bin_edges.size - 1)
        # self.binMid = np.empty(bin_edges.size - 1)
        #
        # for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        #     data_in_bin = self.dat_sorted[(self.rad_sorted >= bin_start) * (self.rad_sorted < bin_end)]
        #     if np.any(data_in_bin):
        #         self.binMin[i], self.binMax[i], self.binMid[i] = np.percentile(data_in_bin, [2, 95, 50])
        #     else:
        #         self.binMin[i], self.binMax[i], self.binMid[i] = np.nan, np.nan, np.nan

        # Bin the intensities by radius
        # self.radial_bins = np.empty((self.rez, self.rez))
        # self.radial_bins.fill(np.nan)
        # radial_counter = np.zeros(self.rez, dtype=np.int)
        # proper_bin = np.asarray(np.floor(self.rad_sorted), dtype=np.int32)

        # for item, binInd in enumerate(proper_bin):
        #     self.radial_bins[binInd, radial_counter[binInd]] = self.dat_sorted[item]
        #     radial_counter[binInd] += 1

        # for ii in np.arange(self.rez):
        #     useItems = np.isin(proper_bin, )
        # self.binMin , self.binMax, self.binMid = self.nan_percentile(self.radial_bins, [2, 95, 50], axis=1)

    # def bin_prep(self):
    #     self.proper_bin = np.asarray(np.floor(self.rad_sorted), dtype=np.int32)
    #
    #     self.binBtm = np.empty(self.rez)
    #     self.binMin = np.empty(self.rez)
    #     self.binMax = np.empty(self.rez)
    #     self.binMid = np.empty(self.rez)
    #     self.binTop = np.empty(self.rez)
    #
    #     self.binBtm.fill(np.nan)
    #     self.binMin.fill(np.nan)
    #     self.binMax.fill(np.nan)
    #     self.binMid.fill(np.nan)
    #     self.binTop.fill(np.nan)

    # def bin_stats(self):
    #     self.bin_prep()
    #
    #     bin_list = []
    #     last = 0
    #     for ii, (binInd, dat) in enumerate(zip(self.proper_bin, self.dat_sorted)):
    #         if binInd != last:
    #             self.bin_the_slice(last, bin_list)
    #             bin_list = []
    #         bin_list.append(dat)
    #         last = binInd

    def bin_the_slice(self, bin_count, bin_list):
        bin_array = np.asarray(bin_list)
        finite = bin_array[np.isfinite(bin_array)]
        data_in_bin = finite[np.nonzero(finite - self.offset)]
        if len(data_in_bin) > 0:
            self.binMin[bin_count], self.binMax[bin_count], self.binMid[bin_count] = np.percentile(data_in_bin, [2, 95, 50])

    def nan_percentile2(self, arr, q, interpolation='linear'):
        # valid (non NaN) observations along the first axis
        valid_obs = np.sum(np.isfinite(arr))
        if valid_obs <= 0:
            return np.nan, np.nan, np.nan
        # replace NaN with maximum
        max_val = np.nanmax(arr)
        arr[np.isnan(arr)] = max_val
        # sort - former NaNs will move to the end
        arr = np.sort(arr)

        # loop over requested quantiles
        if type(q) is list:
            qs = []
            qs.extend(q)
        else:
            qs = [q]
        if len(qs) < 2:
            quant_arr = np.zeros(shape=(arr.shape[0]))
        else:
            quant_arr = np.zeros(shape=(len(qs), arr.shape[0]))

        result = []
        for i in range(len(qs)):
            quant = qs[i]
            # desired position as well as floor and ceiling of it
            k_arr = (valid_obs - 1) * (quant / 100.0)
            f_arr = np.floor(k_arr).astype(np.int32)
            c_arr = np.ceil(k_arr).astype(np.int32)
            fc_equal_k_mask = f_arr == c_arr

            if interpolation == 'linear':
                # linear interpolation (like numpy percentile) takes the fractional part of desired position
                floor_val = np.take(arr, f_arr) * (c_arr - k_arr)
                ceil_val = np.take(arr, c_arr) * (k_arr - f_arr)

                quant_arr = floor_val + ceil_val
                if fc_equal_k_mask:
                    quant_arr = np.take(arr, k_arr.astype(np.int32))  # if floor == ceiling take floor value


            elif interpolation == 'nearest':
                f_arr = np.around(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'lowest':
                f_arr = np.floor(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'highest':
                f_arr = np.ceiling(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            result.append(quant_arr)

        return result

    def nan_percentile(self, arr, q, interpolation='linear', axis=0):
        # valid (non NaN) observations along the first axis
        valid_obs = np.sum(np.isfinite(arr), axis=axis)
        # replace NaN with maximum
        max_val = np.nanmax(arr)
        arr[np.isnan(arr)] = max_val
        # sort - former NaNs will move to the end
        arr = np.sort(arr, axis=axis)

        # loop over requested quantiles
        if type(q) is list:
            qs = []
            qs.extend(q)
        else:
            qs = [q]
        if len(qs) < 2:
            quant_arr = np.zeros(shape=(arr.shape[0], arr.shape[1]))
        else:
            quant_arr = np.zeros(shape=(len(qs), arr.shape[0], arr.shape[1]))

        result = []
        for i in range(len(qs)):
            quant = qs[i]
            # desired position as well as floor and ceiling of it
            k_arr = (valid_obs - 1) * (quant / 100.0)
            f_arr = np.floor(k_arr).astype(np.int32)
            c_arr = np.ceil(k_arr).astype(np.int32)
            fc_equal_k_mask = f_arr == c_arr

            if interpolation == 'linear':
                # linear interpolation (like numpy percentile) takes the fractional part of desired position
                floor_val = np.take(arr, f_arr) * (c_arr - k_arr)
                ceil_val = np.take(arr, c_arr) * (k_arr - f_arr)

                quant_arr = floor_val + ceil_val
                quant_arr[fc_equal_k_mask] = np.take(arr, k_arr.astype(np.int32))[fc_equal_k_mask]

            elif interpolation == 'nearest':
                f_arr = np.around(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'lowest':
                f_arr = np.floor(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'highest':
                f_arr = np.ceiling(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)

            result.append(quant_arr)

        return result

    def fit_curves(self):
        # Input Stuff
        self.radAbss = np.arange(self.rez)
        self.highCut = 730 * self.extra_rez
        theMin = 380 * self.extra_rez
        near_limb = np.arange(theMin, theMin + 50 * self.extra_rez)

        # Find the derivative of the binned Mid
        self.diff_Mid = np.diff(self.binMid)
        self.diff_Mid += np.abs(np.nanmin(self.diff_Mid))
        self.diff_Mid /= np.nanmean(self.diff_Mid) / 100

        # Locate the Limb
        self.limb_radii = np.argmin(self.diff_Mid[near_limb]) + theMin
        self.lCut = self.limb_radii - 10 * self.extra_rez
        self.hCut = self.limb_radii + 10 * self.extra_rez

        # Split into three regions
        self.low_abs = self.radAbss[:self.lCut]
        self.low_max = self.binMax[:self.lCut]
        self.low_min = self.binMin[:self.lCut]

        self.mid_abs = self.radAbss[self.lCut:self.hCut]
        self.mid_max = self.binMax[self.lCut:self.hCut]
        self.mid_min = self.binMin[self.lCut:self.hCut]

        self.high_abs = self.radAbss[self.hCut:]
        self.high_max = self.binMax[self.hCut:]
        self.high_min = self.binMin[self.hCut:]

        # Filter the regions separately
        from scipy.signal import savgol_filter

        lWindow = 20 * self.extra_rez + 1
        mWindow = 4 * self.extra_rez + 1
        hWindow = 30 * self.extra_rez + 1
        fWindow = int(3 * self.extra_rez) + 1

        rank = 3

        low_max_filt = savgol_filter(self.low_max, lWindow, rank)

        mid_max_filt = savgol_filter(self.mid_max, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)

        high_max_filt = savgol_filter(self.high_max, hWindow, rank)

        low_min_filt = savgol_filter(self.low_min, lWindow, rank)
        mid_min_filt = savgol_filter(self.mid_min, mWindow, rank)
        high_min_filt = savgol_filter(self.high_min, hWindow, rank)

        # Fit the low curves
        lmaxf = self.fill_start(low_max_filt)
        lminf = self.fill_start(low_min_filt)
        idx = np.isfinite(lmaxf)
        p = np.polyfit(self.low_abs[idx], lmaxf[idx], 9)
        low_max_fit = np.polyval(p, self.low_abs)
        p = np.polyfit(self.low_abs[idx], lminf[idx], 9)
        low_min_fit = np.polyval(p, self.low_abs)

        # Build output curves
        self.fakeAbss = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.fakeMax = np.hstack((low_max_fit, mid_max_filt, high_max_filt))
        self.fakeMin = np.hstack((low_min_fit, mid_min_filt, high_min_filt))

        # Filter again to smooth boundaraies
        self.fakeMax = self.fill_end(self.fill_start(savgol_filter(self.fakeMax, fWindow, rank)))
        self.fakeMin = self.fill_end(self.fill_start(savgol_filter(self.fakeMin, fWindow, rank)))

        # Locate the Noise Floor
        noiseMin = 550 * self.extra_rez - self.hCut
        near_noise = np.arange(noiseMin, noiseMin + 100 * self.extra_rez)
        self.diff_max_abs = self.high_abs[near_noise]
        self.diff_max = np.diff(high_max_filt)[near_noise]
        self.diff_max += np.abs(np.nanmin(self.diff_max))
        self.diff_max /= np.nanmean(self.diff_max) / 100
        self.noise_radii = np.argmin(self.diff_max) + noiseMin + self.hCut
        self.noise_radii = 565 * self.extra_rez

    def fill_end(self, use):
        iii = -1
        val = use[iii]
        while np.isnan(val):
            iii -= 1
            val = use[iii]
        use[iii:] = val
        return use

    def fill_start(self, use):
        iii = 0
        val = use[iii]
        while np.isnan(val):
            iii += 1
            try:
                val = use[iii]
            except:
                return use
        use[:iii] = val
        return use

    def plot_stats(self):

        fig, (ax0, ax1) = plt.subplots(2, 1, True)
        ax0.scatter(self.n2r(self.rad_sorted[::30]), self.dat_sorted[::30], c='k', s=2)
        ax0.axvline(self.n2r(self.limb_radii), ls='--', label="Limb")
        ax0.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        ax0.axvline(self.n2r(self.lCut), ls=':')
        ax0.axvline(self.n2r(self.hCut), ls=':')
        # ax0.axvline(self.tRadius, c='r')
        ax0.axvline(self.n2r(self.highCut))

        # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
        # plt.plot(self.radAbss[:-1] + 0.5, self.diff_Mid, 'r:')

        ax0.plot(self.n2r(self.low_abs), self.low_max, 'm', label="Percentile")
        ax0.plot(self.n2r(self.low_abs), self.low_min, 'm')
        # plt.plot(self.low_abs, self.low_max_fit, 'r')
        # plt.plot(self.low_abs, self.low_min_fit, 'r')

        ax0.plot(self.n2r(self.high_abs), self.high_max, 'c', label="Percentile")
        ax0.plot(self.n2r(self.high_abs), self.high_min, 'c')

        ax0.plot(self.n2r(self.mid_abs), self.mid_max, 'y', label="Percentile")
        ax0.plot(self.n2r(self.mid_abs), self.mid_min, 'y')
        # plt.plot(self.high_abs, self.high_min_fit, 'r')
        # plt.plot(self.high_abs, self.high_max_fit, 'r')

        try:
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMin, 'g')
        except:
            ax0.plot(self.n2r(self.radAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.radAbss), self.fakeMin, 'g')

        # plt.plot(radAbss, binMax, 'c')
        # plt.plot(self.radAbss, self.binMin, 'm')
        # plt.plot(self.radAbss, self.binMid, 'y')
        # plt.plot(radAbss, binMed, 'r')
        # plt.plot(self.radAbss, self.binMax, 'b')
        # plt.plot(radAbss, fakeMin, 'r')
        # plt.ylim((-100, 10**3))
        # plt.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        ax0.set_xlim((0, self.n2r(self.highCut)))
        ax0.legend()
        fig.set_size_inches((8, 12))
        ax0.set_yscale('log')

        ax1.scatter(self.n2r(self.rad_flat[::10]), self.dat_coronagraph[::10], c='k', s=2)
        ax1.set_ylim((-0.25, 2))

        ax1.axhline(self.vmax, c='r', label='Confinement')
        ax1.axhline(self.vmin, c='r')
        ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
        ax1.axhline(self.vmin_plot, c='orange')

        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))

        ax1.legend()
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax0.set_ylabel(r"Absolute Intensity (Counts)")

        plt.tight_layout()
        if self.params.is_debug():
            file_name = '{}_Radial.png'.format(self.name)
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)

            file_name = '{}_Radial_zoom.png'.format(self.name)
            ax0.set_xlim((0.9, 1.1))
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)
            # plt.show()
            plt.close(fig)
        else:
            plt.show()

    def n2r(self, n):
        if True:
            return n / self.limb_radii
        else:
            return n


class Sonifier:
    def __init__(self, params, save_path, name, vid_stem, bpm=None, scale=None, frames_per_second=None):
        self.params = params
        self.name = name
        self.scale = scale
        self.video_name_stem = vid_stem
        self.first_frame = True
        self.init_speeds(bpm, frames_per_second)
        self.init_paths(save_path)
        self.init_instruments()

    def init_paths(self, save_path):
        self.save_path = save_path
        self.midi_path = join(self.save_path, "{}.mid".format(self.name))
        self.wav_path = join(self.save_path, "{}.wav".format(self.name))
        self.score_path = join(self.save_path, "{}.score".format(self.name))

    def init_speeds(self, bpm, frames_per_second):
        self.skip=1
        if frames_per_second is None:
            frames_per_second = self.params.frames_per_second()
        self.frames_per_second = frames_per_second
        if bpm is None:
            bpm = self.params.bpm()
        self.bpm = bpm
        self.beats_per_second = bpm/60
        self.seconds_per_beat = 1/self.beats_per_second
        self.frames_per_beat = self.frames_per_second * self.seconds_per_beat

    def note_length(self, note):
        """ Takes in a note type (1,2,4,8,16,32) and returns its duration in seconds"""
        mult = 2 ** -(np.log(note) / np.log(2) - 2)
        return np.round(self.seconds_per_beat * mult, 6)

    def skip_frames(self, sec):
        """Returns the number of frames corresponding to a given number of seconds"""
        return int(np.round(sec*self.frames_per_second, 0))

    def frame_time(self, frames):
        """returns the number of seconds corresponding to a frame"""
        return frames/self.frames_per_second

    def frame_on_beat(self, ind, note=None, sec=None, skip=None):
        if note is not None:
            sec = self.note_length(note)
        if sec is not None:
            skip = self.skip_frames(sec)

        sec = self.frame_time(skip)
        go = np.mod(ind, skip) == 0
        # print("Note: {}, Skip {}, ind: {}, sec= {:0.3f}, go = {}".format(note, skip, ind, sec, go))

        if go:
            return sec
        else:
            return False

    def frame_on_any_beat(self, ind):
        for ii in np.arange(6):
            if self.frame_on_beat(ind, 2**ii):
                return True
        return False

    def init_instruments(self):
        """Initialize all the instruments for this score"""
        self.song = dsp.buffer()
        self.instruments = []
        # self.instruments.append(MaxBeeper(self))
        # self.instruments.append(MaxBeeperSliceLeft(self))
        # self.instruments.append(MaxBeeperSliceRight(self))
        self.instruments.append(Segmentor(self))

    def remove_stats(self, data):
        array = data[:-5, :]
        bottom = data[-5, :]
        min = data[-4, :]
        mid = data[-3, :]
        max = data[-2, :]
        top = data[-1, :]
        return array, (bottom, min, mid, max, top)

    def sonify_frame(self, processed_image_stats, raw_image, file_idx):
        """Create a phrase for each instrument using this input"""
        for inst in self.instruments:
            if self.first_frame:
                inst.init_frame(raw_image)
            inst.sonify_image(file_idx, processed_image_stats, raw_image)
        self.first_frame=False

    def generate_track(self, path=None):
        """Write the song to file"""
        if path is None:
            path = self.wav_path

        # if not self.params.allow_muxing():
        self.create(path)

    def create(self, wav):
        self.instruments[0].create(wav)

    def play(self):
        """Play the Generated Sound File"""
        ps(self.wav_path)

    def thread_lock(self):
        self.instruments[0].thread_lock()

class Instrument:
    def __init__(self, soni):
        """Define the way this instrument should sound"""
        self.soni = soni
        self.wav_path = soni.wav_path
        self.frames_per_second = soni.frames_per_second
        self.song = soni.song
        self.counter_dict = dict()
        self.counter=np.zeros(10)

    def play(self):
        """Play the Generated Sound File"""
        ps(self.wav_path)

    def init_frame(self, data):

        self.rez = data.shape[0]
        self.rezX, self.rezY = data.shape
        centerPt = self.rez / 2
        xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
        xc, yc = xx - centerPt, yy - centerPt

        self.extra_rez = 1

        self.sRadius = 400 * self.extra_rez
        self.tRadius = self.sRadius * 1.28
        self.bRadius = self.sRadius * 1.01
        self.radius = np.sqrt(xc * xc + yc * yc) * self.extra_rez
        self.theta = np.arctan2(xc,yc)
        self.rez *= self.extra_rez
        self.on_disk = self.radius <= self.bRadius
        self.off_limb = self.radius > self.bRadius
        self.xx, self.yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))

        # Create arrays sorted by radius
        self.rad_flat = self.radius.flatten()
        self.inds = np.argsort(self.rad_flat)
        self.rad_sorted = self.rad_flat[self.inds]
        self.dat_flat = data.flatten()
        self.dat_sorted = self.dat_flat[self.inds]

        self.video_avi = cv2.VideoWriter(self.soni.video_name_stem.format("_son.avi"), cv2.VideoWriter.fourcc("H", "2", "6", "4"), self.frames_per_second,
                                    (data.shape[0], data.shape[1]))

    def sort_flat(self, data):
        dat_sorted = data.flatten()[self.inds]
        return self.rad_sorted, dat_sorted

    def sonify_image(self, frame_ind, processed_image_stats, raw_image):
        """Generate a phrase from the given input"""
        proc_im, stats = self.soni.remove_stats(processed_image_stats)
        self.frame_ind = frame_ind
        self.voice(frame_ind, proc_im, raw_image, stats)

    def get_mask(self, dat_out, kind=None):
        if kind is None:
            kind = self.kind
        mask = np.full_like(dat_out, True, dtype=bool)
        lenX, lenY = mask.shape
        halfX, halfY = int(lenX/2), int(lenY/2)

        if 'l' in kind:
            mask[halfX:, :] = False
        if 'r' in kind:
            mask[:halfX, :] = False
        if 'u' in kind:
            mask[:, :halfY] = False
        if 'd' in kind:
            mask[:, halfY:] = False

        mask = np.invert(mask)
        out = dat_out + 0
        out[mask] = np.nan
        return mask, out

    def maxChop(self, array, num, wid=75, plotColor=None, plot=False):
        """Returns the loc and value of the num highest values in an array"""
        maxes = []

        new_array = array + 0
        for ii in np.arange(num):
            maxInd = np.nanargmax(new_array)
            maxes.append((maxInd, array[maxInd]))
            low, high = np.max((maxInd - wid, 0)), np.min((maxInd + wid,len(array)))
            new_array[low:high] = np.nan

        if plot:
            for loc, val in maxes:
                plt.axvline(loc, c=plotColor)
                plt.scatter(loc, val, c=plotColor)
        return maxes

    def parse_kind(self, kind):
        if type(kind) in [int, float]:
            return kind
        if type(kind) not in [str]:
            raise TypeError
        if kind.casefold() in 'SINE    '.casefold(): return  0
        if kind.casefold() in 'SINEIN  '.casefold(): return  17
        if kind.casefold() in 'SINEOUT '.casefold(): return  18
        if kind.casefold() in 'COS     '.casefold(): return  1
        if kind.casefold() in 'TRI     '.casefold(): return  2
        if kind.casefold() in 'SAW     '.casefold(): return  3
        if kind.casefold() in 'RSAW    '.casefold(): return  4
        if kind.casefold() in 'HANN    '.casefold(): return  5
        if kind.casefold() in 'HANNIN  '.casefold(): return  21
        if kind.casefold() in 'HANNOUT '.casefold(): return  22
        if kind.casefold() in 'HAMM    '.casefold(): return  6
        if kind.casefold() in 'BLACK   '.casefold(): return  7
        if kind.casefold() in 'BLACKMAN'.casefold(): return  7
        if kind.casefold() in 'BART    '.casefold(): return  8
        if kind.casefold() in 'BARTLETT'.casefold(): return  8
        if kind.casefold() in 'KAISER  '.casefold(): return  9
        if kind.casefold() in 'SQUARE  '.casefold(): return  10
        if kind.casefold() in 'RND     '.casefold(): return  11
        if kind.casefold() in 'LINE    '.casefold(): return  3 # SAW
        if kind.casefold() in 'PHASOR  '.casefold(): return  3 # SAW
        if kind.casefold() in 'SINC    '.casefold(): return  23
        if kind.casefold() in 'GAUSS   '.casefold(): return  24
        if kind.casefold() in 'GAUSSIN '.casefold(): return  25
        if kind.casefold() in 'GAUSSOUT'.casefold(): return  26
        if kind.casefold() in 'PLUCKIN '.casefold(): return  27
        if kind.casefold() in 'PLUCKOUT'.casefold(): return  28
        if kind.casefold() in 'LINEAR  '.casefold(): return  12
        if kind.casefold() in 'TRUNC   '.casefold(): return  13
        if kind.casefold() in 'HERMITE '.casefold(): return  14
        if kind.casefold() in 'CONSTANT'.casefold(): return  15
        if kind.casefold() in 'GOGINS  '.casefold(): return  16

    def parse_adsr(self, adsr, dur):
        if adsr is None:
            adsr = [dur / 10, 0, 1, dur / 10]
        elif type(adsr) in [int, float]:
            adsr = [dur / adsr, 0, 1, dur / adsr]
        return [np.round(it,4) for it in adsr]

    def record_note(self, note, note_props=None, frame_ind=None, delay=0):
        """Record input note into the song"""
        if note_props is not None:
            frame_ind, freq, amp, dur, pan, kind, delay, adsr, beat, *mods = note_props

        self.song.dub(note, delay + (frame_ind / self.frames_per_second))
        if beat not in self.counter_dict:
            self.counter_dict[beat] = 0
        self.counter_dict[beat] += 1
        return note, note_props

    def make_note_props(self, frame_ind=0, freq=440, amp=0.5, dur=1., pan=0.5, kind=0, delay=0, adsr=0, beat=0, *mods):
        """Concatenate note properties into a list"""
        return [frame_ind, np.round(freq, 5), np.round(amp, 5), np.round(dur, 6), np.round(pan, 5), int(kind), np.round(delay, 5), adsr, str(beat), *mods]

    def record_osc_note(self, freq=440, amp=0.5, dur=1., pan=0.5, kind=0, delay=0, adsr=None, beat=0, *mods):
        """Generate a note and record it into the song"""
        return self.record_note(*self.make_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat, *mods))

    def split_list(self, notes, nGroups=4):
        nNotes = len(notes)
        groups = int(nNotes/nGroups)
        box=[]
        for ii in np.arange(nGroups):
            lowInd = ii*groups
            highInd = np.min(((ii+1)*groups, len(notes)))
            box.append(notes[lowInd:highInd])
        return box

    def keep_in_range(self, freq, low, high):
        while freq >= high:
            freq /= 2
        while freq < low:
            freq *= 2
        return freq

    def make_osc_note(self, freq=440, amp=0.5, dur=1., pan=0.5, kind=0, delay=0, adsr=None, beat=0, *mods):
        """Generate a note"""
        # Input Cleaning
        amp = 1 if amp > 1 else 0 if amp < 0 else amp
        pan = 1 if pan > 1 else 0 if pan < 0 else pan
        kind = self.parse_kind(kind)

        # Tone Generation
        oscA = Osc(kind, freq=freq, amp=amp)
        note = oscA.play(dur)
        note.pan(pan)

        # Effects
        adsr = self.parse_adsr(adsr, dur)
        note = note.adsr(*adsr)

        return note, self.make_note_props(self.frame_ind, freq, amp, dur, pan, kind, delay, adsr, beat, *mods)

    def thread_lock(self):
        for th in self.threads:
            th.join()

class MaxBeeper(Instrument):
    def __init__(self, soni):
        super().__init__(soni)

    def effects(self, note, note_props):
        frame_ind, freq, amp, dur, pan, kind, delay, *mods = note_props

        note.adsr(
            a=dsp.rand(0.05 * dur, 0.2 * dur),  # Attack between 50ms and 200ms
            d=dsp.rand(0.1 * dur, 0.3 * dur),  # Decay between 100ms and 300ms
            s=dsp.rand(0.2 * dur, 0.6 * dur),  # Sustain between 10% and 50%
            r=dsp.rand(1 * dur, 2 * dur)  # Release between 1 and 2 seconds*)
        )

    def voice(self, frame_ind, processed_image, raw_image):
        if np.mod(frame_ind, self.skip) == 0:
            return
        maxX, maxY = np.unravel_index(np.nanargmax(processed_image), processed_image.shape)
        theMid = np.mean(raw_image)
        brightest = raw_image[maxX, maxY]

        freq = maxY
        amp = np.min((1, brightest / theMid / 8))
        dur = dsp.rand(1, 1.5) * self.skip / self.frames_per_second
        pan = np.max((0, np.min((1, maxX / 1024))))
        kind = 0
        delay = 0

        self.add_note(freq, amp, dur, pan, kind, delay)
        self.add_note(freq/2, amp/1.5, dur*2, 1-pan, kind+1, delay+0.5)

class MaxBeeperSliceLeft(Instrument):
    def __init__(self, soni):
        super().__init__(soni)
        self.kind='l'
        self.skip = 10

    def effects(self, note, note_props):
        frame_ind, freq, amp, dur, pan, kind, delay, *mods = note_props

        note.adsr(
            a=dsp.rand(0.05 * dur, 0.2 * dur),  # Attack between 50ms and 200ms
            d=dsp.rand(0.1 * dur, 0.3 * dur),  # Decay between 100ms and 300ms
            s=dsp.rand(0.5 * dur, 0.9 * dur),  # Sustain between 10% and 50%
            r=dsp.rand(1 * dur, 2 * dur)  # Release between 1 and 2 seconds*)
        )

    def voice(self, frame_ind, processed_image, raw_image):
        if np.mod(frame_ind, self.skip) == 0:
            return
        mask, use_box = self.get_mask(processed_image, 'lu')

        maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
        theMean = np.mean(raw_image)
        brightest = raw_image[maxX, maxY]

        freq = maxY
        amp = np.min((1, brightest / theMean / 8))
        dur = dsp.rand(1, 1.5) * self.skip / self.frames_per_second
        pan = 0.5*maxX / 1024
        kind = 2
        delay = 0

        self.add_note(freq, amp, dur, pan, kind, delay)

        mask, use_box = self.get_mask(processed_image, 'ld')
        maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
        theMean = np.mean(raw_image)
        brightest = raw_image[maxX, maxY]

        freq = maxY
        amp = np.min((1, brightest / theMean / 8))
        dur = 3*dsp.rand(1, 1.5) * self.skip / self.frames_per_second
        pan = 0.5*maxX / 1024
        kind = 3
        delay = 0.5

        self.add_note(freq, amp, dur, pan, kind, delay)

class MaxBeeperSliceRight(Instrument):
    def __init__(self, soni):
        super().__init__(soni)
        self.kind='r'
        self.skip=5

    def effects(self, note, note_props):
        frame_ind, freq, amp, dur, pan, kind, delay, *mods = note_props

        note.adsr(
            a=dsp.rand(0.05 * dur, 0.2 * dur),  # Attack between 50ms and 200ms
            d=dsp.rand(0.1 * dur, 0.3 * dur),  # Decay between 100ms and 300ms
            s=dsp.rand(0.1 * dur, 0.5 * dur),  # Sustain between 10% and 50%
            r=dsp.rand(1 * dur, 2 * dur)  # Release between 1 and 2 seconds*)
        )

    def voice(self, frame_ind, processed_image, raw_image):
        if np.mod(frame_ind, self.skip) == 0:
            return
        mask, use_box = self.get_mask(processed_image, 'ru')

        maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
        theMean = np.mean(raw_image)
        brightest = raw_image[maxX, maxY]

        freq = maxY
        amp = np.min((1, brightest / theMean / 8))
        dur = dsp.rand(1, 1.5) * self.skip / self.frames_per_second
        pan = (maxX / 1024)/2 + 0.5
        kind = 0
        delay = 0

        self.add_note(freq, amp, dur, pan, kind, delay)

        mask, use_box = self.get_mask(processed_image, 'rd')
        maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
        theMean = np.mean(raw_image)
        brightest = raw_image[maxX, maxY]

        freq = maxY / 2
        amp = np.min((1, brightest / theMean / 8))
        dur = 3 * dsp.rand(1, 1.5) * self.skip / self.frames_per_second
        pan = (maxX / 1024)/2 + 0.5
        kind = 1
        delay = 0.5

        self.add_note(freq, amp, dur, pan, kind, delay)

from proglog import TqdmProgressBarLogger


class Segmentor(Instrument):
    def __init__(self, soni):
        super().__init__(soni)
        from collections import defaultdict
        self.strain = defaultdict(lambda :True)
        self.strain8 = False
        self.strain4 = False
        self.ax=None
        self.tries = 0
        self.cutoff = 127
        self.max_tries = 20
        self.mean_adjust=-60
        self.imbox = []

        self.chord = tune.next_chord("I")

    def create(self, path):
        # print(">Writing Sound...", end="")
        print(self.counter_dict)
        self.sound_writer(path)
        self.movie_writer()
        # print("Success!")

        # Thread(target=self.movie_writer).start()
        # Thread(target=self.play).start()

    def play_movie(self):
        bbb.wait()
        # startfile(self.soni.video_name_stem.format("_son.mp4"))

    def write_wrapper_son_hq(self, videoclip_full_muxed):
        videoclip_full_muxed.write_videofile(self.soni.video_name_stem.format("_son.mp4"), codec='libx264', bitrate='200M',
                                             logger=TqdmProgressBarLogger(print_messages=False))
        bbb.wait()
    def write_wrapper_son_lq(self, videoclip_full_muxed):
        bbb.wait()
        print("Starting")
        videoclip_full_muxed.write_videofile(self.soni.video_name_stem.format("_son_lq.mp4"), codec='libx264', bitrate='5M',
                                             logger=TqdmProgressBarLogger(print_messages=False))
        print("Done")
    def movie_writer(self):

        cv2.destroyAllWindows()
        self.video_avi.release()

        videoclip_full = VideoFileClip(self.soni.video_name_stem.format("_son.avi"))
        videoclip_full_muxed = videoclip_full.set_audio(AudioFileClip(self.wav_path))

        hq_sonFunc = partial(self.write_wrapper_son_hq, videoclip_full_muxed)
        lq_sonFunc = partial(self.write_wrapper_son_lq, videoclip_full_muxed)

        t1 = Thread(target=hq_sonFunc)
        t2 = Thread(target=lq_sonFunc)
        t3 = Thread(target=self.play_movie)
        t1.start()
        t2.start()
        t3.start()
        self.threads = [t1,t2,t3]
        # videoclip_full_muxed.write_videofile(self.soni.video_name_stem.format("_son.mp4"), codec='libx264', bitrate='200M',
        #                                      logger=TqdmProgressBarLogger(print_messages=False))
        # videoclip_full_muxed.write_videofile(self.soni.video_name_stem.format("_son_lq.mp4"), codec='libx264', bitrate='5M',
        #                                      logger=TqdmProgressBarLogger(print_messages=False))
        # for ii in np.arange(3):
        #     try:
        #         remove(self.soni.video_name_stem.format("_son.avi"))
        #     except:
        #         continue
        #     break

    def sound_writer(self, path):
        self.wav_path = path
        fx.norm(self.song, 1)
        self.song.write(self.wav_path)


    def radial_notes(self, stats):
        btm, mina, mida, maxa, top = stats

        # rad_sort1, dat_sort = self.sort_flat(raw_image)
        # rad_sort2, dat_sort_proc = self.sort_flat(processed_image)
        #
        # theMaxHere=np.max(maxa)
        # dat_sort_proc_tall = [theMaxHere*x for x in dat_sort_proc]
        # plt.figure()
        # plt.plot(aStat, btm, 'darkred')
        # plt.plot(aStat, mina, 'r')
        # plt.plot(aStat, mida, 'g')
        # plt.plot(aStat, maxa, 'b')
        # plt.plot(aStat, top, 'darkblue')
        # plt.scatter(rad_sort1, dat_sort, c='k')
        # # plt.scatter(rad_sort2, dat_sort_proc_tall, c='lightgrey')
        # plt.show()

        low = mina/btm
        med = mida/mina
        high = maxa/mida
        vhigh = top/maxa

        doPlot = False

        if doPlot:
            plt.figure()
            aStat = np.arange(len(mina))
            plt.plot(aStat, low, 'g')
            plt.plot(aStat, med, 'r')
            plt.plot(aStat, high, 'b')
            plt.plot(aStat, vhigh, 'y')
        notes_low   = self.maxChop(low, 2, 150, 'g', doPlot)
        notes_med   = self.maxChop(med, 2, 125, 'r', doPlot)
        notes_high  = self.maxChop(high, 4, 75, 'b', doPlot)
        notes_vhigh = self.maxChop(vhigh, 2, 50,'y', doPlot)

        all_notes = notes_low + notes_med + notes_high + notes_vhigh
        all_freqs = [note[0] for note in all_notes]
        sortInds = np.argsort(all_freqs)
        sorted_notes = [all_notes[ind] for ind in sortInds]

        if doPlot:
            plt.yscale('log')
            plt.show()
        return sorted_notes


    def array2uint_proc(self, image):
        maxa = 1.9 #np.nanmax(image)
        mina = 0.06 #np.nanmin(image)
        rescaled = 255 * (image - mina) / (maxa - mina)
        return np.abs(rescaled).astype('uint8')

    def array2uint(self, image):
        maxa = np.nanmax(image)
        mina = np.nanmin(image)
        rescaled = 255 * (image - mina) / (maxa - mina)
        return np.abs(rescaled).astype('uint8')

    def grey2clr(self, img_grey):
        themap = 255 * cm.viridis(img_grey)[:, :, :-1]
        img_clr = np.abs(themap)
        # img_clr = img_clr[:, :, [2,1,0]]
        img_clr = np.flip(img_clr, axis=2).astype('uint8')
        return img_clr

    def get_regions(self, image, where='both'):
        import imutils

        # Get Image
        # img_grey=self.array2uint_proc(image)
        img_grey=self.array2uint(image)
        # cv2.cvtColor(image, cv2.)
        # Remove unimportant regions
        if where in "on disk":
            img_grey[self.off_limb] = np.mean(img_grey[self.off_limb])
        elif where in "off limb":
            img_grey[self.on_disk] = np.mean(img_grey[self.on_disk])

        cnt_img = img_grey+0

        # Blur to remove spurious hot pixels
        blurSz = 3
        blurred = cv2.GaussianBlur(img_grey, (blurSz,blurSz), 0)
        blurred = self.array2uint(blurred.astype('float32')*blurred.astype('float32'))
        # self.tries = 0
        # done=False
        # cutoff = 50
        # lowcut = 255
        # highcut = 0
        # foundhi = False
        # foundlo = False
        # print("")
        # while True:
        #
        #     modded = cv2.threshold(blurred, cutoff, 255, cv2.THRESH_BINARY)[1]
        #     # Remove the Noise
        #     ksize = 5
        #     its = 2
        #     modded = cv2.morphologyEx(modded, cv2.MORPH_CLOSE, np.ones((ksize,ksize), np.uint8), iterations=its)
        #     modded = cv2.morphologyEx(modded, cv2.MORPH_OPEN, np.ones((ksize,ksize), np.uint8), iterations=its)
        #
        #     # Label the Markers
        #     ret, markers = cv2.connectedComponents(modded)
        #     ret -= 1 #Don't count the background as a region
        #
        #
        #     print(foundlo, foundhi, cutoff, ret)
        #     self.tries += 1
        #     if self.tries > 100:
        #         sys.exit()
        #     # print(ret)
        #     # import pdb; pdb.set_trace()
        #     if 8 < ret < 25 and not foundlo:
        #         lowcut = cutoff
        #         foundlo = True
        #     elif 0 < ret < 3 and not foundhi:
        #         highcut = cutoff
        #         foundhi = True
        #     elif not foundlo:
        #         cutoff += 10
        #     elif not foundhi:
        #         # if ret > 50:
        #         #     cutoff -= 6
        #         # else:
        #         cutoff -= 8
        #
        #     else:
        #         break
        # highcut = 10
        # lowcut = 200
        # self.tries = 0
        # print(highcut, lowcut)
        # Setup SimpleBlobDetector parameters.

        # Make Detector
        params = cv2.SimpleBlobDetector_Params()
        try:
            params.filterByColor = True
            params.blobColor = 255

            # Change thresholds
            params.minThreshold = 100
            params.maxThreshold = 210
            params.thresholdStep = 5

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 100

            # Filter by Circularity
            params.filterByCircularity = False
            params.minCircularity = 0.0
            params.maxCircularity = 0.5

            # Filter by Convexity
            params.filterByConvexity = False
            params.minConvexity = 0.8
            # params.maxConvexity = 0.99

            # Filter by Inertia
            params.filterByInertia = False
            params.maxInertiaRatio = 0.3

            params.minDistBetweenBlobs = 100

            # Create a detector with the parameters
            ver = (cv2.__version__).split('.')
            if int(ver[0]) < 3:
                detector = cv2.SimpleBlobDetector(params)
            else:
                detector = cv2.SimpleBlobDetector_create(params)
        except:
            raise


        # Detect the points
        use = blurred
        keypoints = detector.detect(use)

        the_notes = []
        for ii, kpt in enumerate(keypoints):
            (cX, cY), sz = kpt.pt, kpt.size

            xc, yc = self.xx - cX, self.yy - cY
            radius = np.sqrt(xc * xc + yc * yc)
            amp = np.sum(use[radius <= sz/2])

            # amp = np.sum(img_grey[markers==ii])

            # M = cv2.moments(c)
            # if M["m00"] == 0:
            #     continue
            # sz = M["m00"]
            # # compute the center of the contour
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            nt = [np.round(cX, 6), np.round(cY,6), amp, np.round(sz,6)]
            # print(nt)
            the_notes.append(nt)

        if len(the_notes) > 0:

            # Sort them
            the_notes = self.sort_notes(the_notes)

            # Throw out the small ones
            the_notes = the_notes[:6]

            # for ii, nt in enumerate(the_notes[:-1]):
            #     if not sz[ii] / sz[0] >= 0.25:
            #         break

            # Plot
            im_with_keypoints = cv2.drawKeypoints(use, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            for (cX, cY, amp, sz) in the_notes:
                center = (int(cX), int(cY))
                # cv2.drawContours(img_clr, [c], -1, (255, 0, 0), 2)
                cv2.circle(im_with_keypoints, center, 6, (0, 0, 255), -1)
                # (x, y), radius = cv2.minEnclosingCircle(c)
                # center = (int(x), int(y))
                # radius = int(radius)
                cv2.circle(im_with_keypoints, center, int(sz), (0, 255, 0), 2)
                # cv2.circle(im_with_keypoints, center, 3, (0,255,0), -1)


            if False:
                if not self.ax:
                    fig, self.ax = plt.subplots(1, 1, True, True)
                    fig.set_size_inches((20, 20))
                    plt.tight_layout()
                # for ii, (x, y, a, s) in enumerate(the_notes):
                #     self.ax.scatter(x, y, 50 * a, 'r' if done else 'w')
                #     self.ax.set_title("Cutoff {:0.4f}, ret {}".format(self.cutoff, ret))
                self.ax.imshow(im_with_keypoints, origin='lower')
                plt.pause(0.5)
                self.ax.cla()
                done = True

            self.video_avi.write(im_with_keypoints)

        return the_notes#, cnts, markers, img_grey

    def sort_notes(self, the_notes, which=3):
        xx, yy, aa, sz  = zip(*the_notes)
        stats = xx, yy, aa, sz
        inds = np.argsort(stats[which])
        the_notes = [the_notes[i] for i in reversed(inds)]
        return the_notes
        # xx, yy, aa, sz = zip(*the_notes)
        # pass








            # cv2.imshow("Keypoints", im_with_keypoints)
            # cv2.waitKey(0)

            # # Threshold the Image
            # thresh = cv2.threshold(blurred, self.cutoff, 255, cv2.THRESH_BINARY)[1]
            # # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, box_size, mean_adjust)
            # # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, box_size, mean_adjust)
            #
            # # Remove the Noise
            # modded = thresh
            # ksize = 5
            # its = 2
            # modded = cv2.morphologyEx(modded, cv2.MORPH_CLOSE, np.ones((ksize,ksize), np.uint8), iterations=its)
            # modded = cv2.morphologyEx(modded, cv2.MORPH_OPEN, np.ones((ksize,ksize), np.uint8), iterations=its)
            #
            # # Label the Markers
            # ret, markers = cv2.connectedComponents(modded)
            # ret -= 1 #Don't count the background as a region
            #
            # # Make Contours
            # cnts = imutils.grab_contours(cv2.findContours(modded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
            #
            # the_notes = []
            # for ii, c in enumerate(cnts):
            #     M = cv2.moments(c)
            #     if M["m00"] == 0:
            #         continue
            #     amp = np.sum(img_grey[markers==ii])
            #     sz = M["m00"]
            #     # compute the center of the contour
            #     cX = int(M["m10"] / M["m00"])
            #     cY = int(M["m01"] / M["m00"])
            #
            #     nt = [cX, cY, amp, sz]
            #     the_notes.append(nt)
            #
            # # Sort them
            # xx, yy, aa, sz = zip(*the_notes)
            # inds = np.argsort(sz)
            # the_notes = [the_notes[i] for i in reversed(inds)]
            # cnts = [cnts[i] for i in reversed(inds)]
            # xx, yy, aa, sz = zip(*the_notes)
            #
            # # Throw out the small ones
            # for ii, nt in enumerate(the_notes[:-1]):
            #     if not sz[ii] / sz[0] >= 0.25:
            #         break
            # the_notes = the_notes[:ii]
            # cnts = cnts[:ii]
            # n_cont = len(cnts)
            #
            # if False:
            #     if not self.ax:
            #         fig, self.ax = plt.subplots(1,1,True,True)
            #         fig.set_size_inches((20,20))
            #     for ii, (x,y,a,s) in enumerate(the_notes):
            #         self.ax.scatter(x,y,50*a, 'r' if done else 'w')
            #         self.ax.set_title("Cutoff {:0.4f}, ret {}".format(self.cutoff, ret))
            #     if done:
            #         self.ax.imshow(img_grey, origin='lower')
            #         plt.pause(1)
            #         self.ax.cla()
            #         self.imbox.append(img_grey)
            #     else:
            #         self.ax.imshow(cnt_img, origin='lower')
            #         plt.pause(0.01)
            #         self.ax.cla()
            #
            #
            # if n_cont < 2:
            #     self.cutoff -= 10
            #     self.mean_adjust +=5
            #     self.tries += 1
            # elif n_cont > 5:
            #     self.cutoff += 8
            #     self.mean_adjust -=4
            #     self.tries += 1
            # else:
            #     done = True
            # if self.tries > self.max_tries:
            #     done = True
            #
            # if done:
            #     # Convert to appropriate type
            #     # img_clr = self.grey2clr(markers/ret*255)
            #     img_clr = self.grey2clr(cnt_img)
            #
            #     for c, (cX, cY, amp, sz) in zip(cnts, the_notes):
            #         cv2.drawContours(img_clr, [c], -1, (255, 0, 0), 2)
            #         cv2.circle(img_clr, (cX, cY), 6, (0, 0, 255), -1)
            #         (x, y), radius = cv2.minEnclosingCircle(c)
            #         center = (int(x), int(y))
            #         radius = int(radius)
            #         cv2.circle(img_clr, center, radius, (0, 255, 0), 2)
            #         cv2.circle(img_clr, center, 3, (0,255,0), -1)



            # self.video_avi.write(img_clr)

            # if False:
            #     if not self.ax:
            #         fig, self.ax = plt.subplots(1,1,True,True)
            #         fig.set_size_inches((20,20))
            #     # for ii, (x, y, a, s) in enumerate(the_notes):
            #     #     self.ax.scatter(x, y, 50 * a, 'r' if done else 'w')
            #     #     self.ax.set_title("Cutoff {:0.4f}, ret {}".format(self.cutoff, ret))
            #     self.ax.imshow(img_clr, origin='lower')
            #     plt.pause(0.1)
            #     self.ax.cla()

            # break


        # self.max_tries = 5

        # return the_notes, cnts, markers, img_grey

    def voice(self, frame_ind, processed_image, raw_image, stats):

        # btm, mina, mida, maxa, top = stats
        #
        # mask, use_box = self.get_mask(processed_image, 'ru')
        #
        # maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
        # theMean = np.mean(raw_image)
        # brightest = raw_image[maxX, maxY]

        # the_notes, cnts, markers, img_grey = self.get_regions(processed_image)
        the_notes = self.get_regions(processed_image)

        if len(the_notes) == 0:
            return

        note_box = self.split_list(the_notes, 4)

        # sorted_radial_notes = self.radial_notes(stats)
        # note_box = self.split_list(sorted_radial_notes, 4)

        # Instrument
        beat = 32
        active = True
        notes = the_notes
        kind = 'cos'  # Waveform
        adsr = 15 # Envelope
        reverb = 1.5  # Duration Multiplier
        jitter = 0.05  # Random offset Seconds
        variance = 0.05  # Duration Percentage
        high_chance = 0.8
        low_chance = 0.1
        keep_all = True
        low_hz = 800
        high_hz = 1200
        limit_hz = True
        amp_factor=0.7
        noteNum = 10

        if self.soni.frame_on_beat(frame_ind, 2):
            # self.chord = tune.next_chord("I")
            chrds = ['I', 'iii', 'V']
            self.chord = chrds[frame_ind%len(chrds)]

        duration = self.soni.frame_on_beat(frame_ind, beat)
        if duration and active and len(notes)>0:
            keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
            self.strain[beat]=False
            if keep:
                self.strain[beat]=True
                delay = dsp.rand(-jitter, jitter)
                dur = duration*reverb*dsp.rand(1-variance, 1+variance)
                # frq_avg = np.mean(all_freq)
                # amp_avg = np.mean(all_amp)

                if not adsr:
                    a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
                    d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
                    s = dsp.rand(0.6, 0.95)  # Sustain between 10% and 50%
                    r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
                    adsr = [a, d, s, r]

                (x0, y0, a0, s0) = notes[0]
                if len(notes)>1:
                    (x1, y1, a1, s1) = notes[1]
                else:
                    (x1, y1, a1, s1) = notes[0]


                for ii, (y, x, a, s) in enumerate(notes):
                    octRange = self.rezY / 4
                    yc = y - self.rezY/2
                    xc = x - self.rezX/2
                    oct = np.round(y / octRange, 0) + 2
                    # flr = oct*octRange
                    # pitch = y % octRange
                    # freqs = []
                    # freqs.extend(tune.chord(self.chord, key='C', octave=oct))
                    # freq = freqs[int(pitch % len(freqs))]

                    # amp = 2*s/(s0+s1) #(s+a)/(s0+a0) #4 * a / (ii + 4) #am/amp_avg * amp_factor #np.min(((am - 1) / 3, 1))
                    amp = 2*a/(a0+a1)
                    # import pdb; pdb.set_trace()
                    r = np.sqrt(xc*xc+yc*yc)
                    z = (r - self.bRadius)/(self.tRadius - self.bRadius)
                    t = np.arctan2(y, x) * 180 / 3.14
                    ct = -np.cos(t)/2 +1.5 # 1 to 2
                    # if r > self.sRadius*0.8:
                    hi_freq = z if z > 0 else 0 # 0 to 1
                    md = 10
                    mid_freq = (t % md)/md # 0 to 1
                    low_freq = (y / self.rezY) # 0 to 1
                    freq = tune.a4*(0.5 + low_freq + 4*mid_freq + 4*hi_freq)/2 # 0-2 # 0-600
                    # freq = (600* + baseline)
                    # print(r, hi_freq, mid_freq, low_freq, freq)
                    # kind = 'cos'
                    # else:
                    #     freq = y
                    #     kind = 'tri'
                    #     amp /= 2

                    # freq = self.keep_in_range(y, low_hz, high_hz) if limit_hz else y

                    # pan = (x - 100)/(1024-200)
                    # pan = 0 if x < 512 else 1
                    # freq = x*2
                    pan = x/self.rezX
                    # if pan < 0.5:
                    #     pan = pan / 6
                    # else:
                    #     pan = pan / 6 + (1-1/6)
                        # continue

                    if ii < noteNum:
                        self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)

        # Instrument
        beat = 8
        active = False
        # notes = note_box[3]
        kind = 'cos'  # Waveform
        adsr = 0 # Envelope
        reverb = 1.25  # Duration Multiplier
        jitter = 0.2  # Random offset Seconds
        variance = 0.15  # Duration Percentage
        high_chance = 0.8
        low_chance = 0.1
        keep_all = True
        low_hz = 700
        high_hz = low_hz * 2
        limit_hz = True
        amp_factor=0.8

        duration = self.soni.frame_on_beat(frame_ind, beat)
        if duration and active:
            keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
            self.strain[beat]=False
            if keep:
                self.strain[beat]=True
                delay = dsp.rand(-jitter, jitter)
                dur = duration*reverb*dsp.rand(1-variance, 1+variance)

                if not adsr:
                    a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
                    d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
                    s = dsp.rand(0.6, 0.95)  # Sustain between 10% and 50%
                    r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
                    adsr = [a, d, s, r]


                freq = maxY #self.keep_in_range(maxY, low_hz, high_hz) if limit_hz else frq
                amp = amp_factor
                pan = maxX/1024

                self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)

        # Instrument
        beat = 8
        active = False
        notes = note_box[2]
        kind = 'cos'  # Waveform
        adsr = 15 # Envelope
        reverb = 1.75  # Duration Multiplier
        jitter = 0.1  # Random offset Seconds
        variance = 0.15  # Duration Percentage
        high_chance = 0.85
        low_chance = 0.15
        keep_all = False
        low_hz = 500
        high_hz = low_hz * 2
        limit_hz = True
        amp_factor=0.7

        duration = self.soni.frame_on_beat(frame_ind, beat)
        if duration and active:
            keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
            self.strain[beat]=False
            if keep:
                self.strain[beat]=True
                delay = dsp.rand(-jitter, jitter)
                dur = duration*reverb*dsp.rand(1-variance, 1+variance)
                all_freq, all_amp = zip(*notes)
                frq_avg = np.mean(all_freq)
                amp_avg = np.mean(all_amp)

                if not adsr:
                    a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
                    d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
                    s = dsp.rand(0.6, 0.95)  # Sustain between 10% and 50%
                    r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
                    adsr = [a, d, s, r]

                for (frq, am) in zip(all_freq, all_amp):
                    freq = self.keep_in_range(frq, low_hz, high_hz) if limit_hz else frq
                    amp = am/amp_avg * amp_factor#np.min(((am - 1) / 3, 1))
                    pan = 1-0.5 * frq/frq_avg

                    self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)

        # Instrument
        beat = 2
        active = False
        notes = note_box[1]
        kind = 'tri'  # Waveform
        adsr = 30 # Envelope
        reverb = 2 # Duration Multiplier
        jitter = 0.0  # Random offset Seconds
        variance = 0.05  # Duration Percentage
        high_chance = 0.95
        low_chance = 0.8
        keep_all = False
        low_hz = 150
        high_hz = 400
        limit_hz = True
        amp_factor = 0.9

        duration = self.soni.frame_on_beat(frame_ind, beat)
        if duration and active:
            keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
            self.strain[beat]=False
            if keep:
                self.strain[beat]=True
                delay = dsp.rand(-jitter, jitter)
                dur = duration*reverb*dsp.rand(1-variance, 1+variance)
                all_freq, all_amp = zip(*notes)
                frq_avg = np.mean(all_freq)
                amp_avg = np.mean(all_amp)

                if not adsr:
                    a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
                    d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
                    s = dsp.rand(0.7 * dur, 0.9 * dur)  # Sustain between 10% and 50%
                    r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
                    adsr = [a, d, s, r]

                for (frq, am) in zip(all_freq, all_amp):
                    freq = self.keep_in_range(frq, low_hz, high_hz) if limit_hz else frq
                    amp = am/amp_avg * amp_factor#np.min(((am - 1) / 3, 1))
                    pan = 1-0.5 * frq/frq_avg

                    self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)

        # Instrument
        beat = 1
        active = False
        notes = note_box[0]
        kind = 'cos'  # Waveform
        adsr = 30 # Envelope
        reverb = 2 # Duration Multiplier
        jitter = 0.0  # Random offset Seconds
        variance = 0.05  # Duration Percentage
        high_chance = 0.98
        low_chance = 0.85
        keep_all = True
        low_hz = 30
        high_hz = 150
        limit_hz = True
        amp_factor = 0.7

        duration = self.soni.frame_on_beat(frame_ind, beat)
        if duration and active:
            keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
            self.strain[beat]=False
            if keep:
                self.strain[beat]=True
                delay = dsp.rand(-jitter, jitter)
                dur = duration*reverb*dsp.rand(1-variance, 1+variance)
                all_freq, all_amp = zip(*notes)
                frq_avg = np.mean(all_freq)
                amp_avg = np.mean(all_amp)

                if not adsr:
                    a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
                    d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
                    s = dsp.rand(0.6, 0.95)  # Sustain between 10% and 50%
                    r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
                    adsr = [a, d, s, r]

                for (frq, am) in zip(all_freq, all_amp):
                    freq = self.keep_in_range(frq, low_hz, high_hz) if limit_hz else frq
                    amp = am/amp_avg * amp_factor#np.min(((am - 1) / 3, 1))
                    pan = 1-0.5 * frq/frq_avg

                    self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)
                    break
# Available to User


def run(delay=20, mode='all', debug=False):
    p = Parameters()
    p.mode(mode)
    p.set_delay_seconds(delay)
    p.do_mirror(False)
    p.do_171(True)

    if debug:
        p.is_debug(True)
        p.set_delay_seconds(10)
        p.do_HMI(False)

    # p.time_period(period=['2019/12/21 04:20', '2019/12/21 04:40'])
    p.resolution(2048)
    p.range(days=5)#0.060)
    p.download_images(True)
    p.cadence(3)
    p.frames_per_second(20)
    p.bpm(150)
    # p.download_images(False)
    # p.overwrite_pngs(False)
    p.sonify_limit(False)
    # p.remove_old_images(True)
    p.make_compressed(True)
    p.sonify_images(True, True)
    # p.sonify_images(False, False)
    # p._stop_after_one = True
    # p.do_171(True)
    # p.do_304(True)

    # Sunback(p).start()
    SunbackMovie(p).start()


def where():
    """Prints the location that the images are stored in."""
    p = Parameters()
    print(p.discover_best_default_directory())


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    where()
    run(20, 'y', debug=debugg)












# cdef int SINE = 0
# cdef int SINEIN = 17
# cdef int SINEOUT = 18
# cdef int COS = 1
# cdef int TRI = 2
# cdef int SAW = 3
# cdef int RSAW = 4
# cdef int HANN = 5
# cdef int HANNIN = 21
# cdef int HANNOUT = 22
# cdef int HAMM = 6
# cdef int BLACK = 7
# cdef int BLACKMAN = 7
# cdef int BART = 8
# cdef int BARTLETT = 8
# cdef int KAISER = 9
# cdef int SQUARE = 10
# cdef int RND = 11
# cdef int LINE = SAW
# cdef int PHASOR = SAW
# cdef int SINC = 23
# cdef int GAUSS = 24
# cdef int GAUSSIN = 25
# cdef int GAUSSOUT = 26
# cdef int PLUCKIN = 27
# cdef int PLUCKOUT = 28