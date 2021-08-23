from os import makedirs, getcwd
from os.path import join, normpath, dirname, abspath, isdir
from time import time, sleep

import numpy as np
from astropy import units as u

from utils.file_util import discover_best_data_directory


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
        self._delay_seconds = 30
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
        self._delete = True
        self._make_compressed = False
        self._remove_old_images = False
        self._sonify_images = True
        self._sonify_limit = True
        self._do_171 = False
        self._do_304 = False
        self._do_one = False
        self._something_changed = False
        self._allow_muxing = True
        
        self._stop_after_one = False
        
        self._time_period = None
        self._range_in_days = 4
        self._cadence = 10 * u.minute
        self._frames_per_second = 30
        self._bpm = 70
        
        self._run_type = "web"
        self._do_one = False
        
        # TODO remove this from params or something
        self._archive_url = None
        self._download_path = None
        self._time_path = None
        self._local_img_paths = None
        self._local_fits_paths = []
        self._fetcher = None
        self._putter = None
        self._processor = []
        self._do_recent = True
        
        self.set_default_values()
    
    # TODO: extract getter/setter logic
    
    def fetcher(self, _fetcher=None):
        if _fetcher is not None:
            self._fetcher = _fetcher
        return self._fetcher
    
    def processors(self, _processor=None):
        if _processor is not None:
            if type(_processor) not in [list]:
                self._processor = [_processor]
            else:
                self._processor = _processor
        return self._processor

    def putter(self, _putter=None):
        if _putter is not None:
            self._putter = _putter
        return self._putter
    
    def archive_url(self, _archive_url=None):
        if _archive_url is not None:
            self._archive_url = _archive_url
        return self._archive_url
 
    def download_path(self, _download_path=None):
        if _download_path is not None:
            self._download_path = _download_path
        return self._download_path
    
    def time_path(self, _time_path=None):
        if _time_path is not None:
            self._time_path = _time_path
        return self._time_path

    def local_fits_paths(self, _local_fits_paths=None):
        if _local_fits_paths is not None:
            self._local_fits_paths = _local_fits_paths
        return self._local_fits_paths
    
    def local_img_paths(self, _local_img_paths=None):
        if _local_img_paths is not None:
            self._local_img_paths = _local_img_paths
        return self._local_img_paths
    
    def run_type(self, _type=None):
        if _type is not None:
            self._run_type = _type
        return self._run_type
    
    def do_one(self, which=False, stop=False):
        if which is not False:
            self._do_one = which
            self.stop_after_one(stop)
        return self._do_one
    
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
    
    def do_recent(self, _do_recent=None):
        if _do_recent is not None:
            assert type(_do_recent) in [bool]
            self._do_recent = _do_recent
        return self._do_recent
    
    def cadence_minutes(self, cad=None):
        if cad is not None:
            self._cadence = cad * u.minute
        return self._cadence
    
    def time_period(self, period=None):
        if period is not None:
            self._time_period = period
            self.do_recent(False)
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
        self.delay_seconds(30 * self.seconds)
        self.set_time_multiplier(3)
        
        # Set File Paths
        self.set_local_directory()
        self.time_file = join(self.local_directory, 'time.txt')
        self.index_file = join(self.local_directory, 'index.txt')
        
        # Set Wavelengths
        self.set_wavelengths(['0171', '0193', '0211', '0304', '0131', '0335', '0094'])
        
        # Set Resolution
        self.set_download_resolution(2048)
        
        # Set Web Location
        self.set_web_image_frame("https://sdo.gsfc.nasa.gov/assets/img/latest/latest_{}_{}")
        
        # # Add extra images
        # new_web_path_1 = "https://sdo.gsfc.nasa.gov/assets/img/latest/f_211_193_171pfss_{}.jpg".format(self.resolution)
        # self.append_to_web_paths(new_web_path_1, 'PFSS')
        
        # Select File Ending
        self.set_file_ending("{}_Now.png")
        
        return 0
    
    def delete_old(self, _delete=None):
        if _delete is not None:
            self._delete = _delete
        return self._delete
    
    def delay_seconds(self, _delay=None):
        if _delay is not None:
            self.check_real_number(_delay)
            self._delay_seconds = _delay
        return self._delay_seconds
    
    # Methods that Set Parameters (LEGACY SETTERS)
    def set_time_multiplier(self, multiplier):
        self.check_real_number(multiplier)
        self.time_multiplier_for_long_display = multiplier
        return 0
    
    def set_local_directory(self, path=None):
        if path is not None:
            self.local_directory = path
        else:
            self.local_directory = discover_best_data_directory()
        
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
    
    
    def determine_delay(self):
        """ Determine how long to wait """
        
        delay = self.delay_seconds + 0
        # import pdb; pdb.set_trace()
        # if 'temp' in wave:
        #     delay *= self.time_multiplier_for_long_display
        
        self.run_time_offset = time() - self.start_time
        delay -= self.run_time_offset
        delay = max(delay, 0)
        return delay
    
    def wait_if_required(self, delay):
        """ Wait if Required """
        
        if delay <= 0:
            pass
        else:
            print("Waiting for {:0.0f} seconds ({} total)".format(delay, self.delay_seconds),
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