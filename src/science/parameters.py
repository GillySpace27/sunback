import os
from os import makedirs, getcwd
from os.path import join, normpath, dirname, abspath, isdir
from time import time, sleep, strftime, asctime

import numpy as np
from astropy import units as u
from fetcher.LocalFetcher import LocalFetcher
from processor.Processor import Processor
from processor.ValidationProcessor import ValidationProcessor
from putter.NullPutter import NullPutter
# from utils.file_util import find_root_directory
import matplotlib.pyplot as plt


class Parameters:
    """
    A container class for the run parameters of the program
    """
    seconds = 1
    minutes = 60 * seconds
    hours = 60 * minutes
    _batch_name = "default"
    
    def __init__(self):
        """Sets all the attributes to None"""
        # Initialize Variables

        self.file_basename = None
        self.orig_path = None
        self.cat_path  = None
        self.mod_path  = None
        self.last_wave = None
        self.analysis_directory = None
        self._imgs_top_directory = None
        self.currently_local = True
        self.norm_curves_name = None
        self._image = None
        self.root_directory = None
        self._shortcut_directory = None
        self.selection = None
        self.paper_out = []
        self.png_frame_name = None
        
        self.tend = ''
        self.tstart = ''
        self._remake_norm_curves = False
        self._params_path = None
        self._curve_path = None
        self._write_video = False
        self._overwrite_pngs = False
        self._reprocess_mode = False
        self._current_wave = 'rainbow'
        self._imgs_directory = None
        self._fits_directory = None
        self._movs_directory = None
        self._base_directory = None
        self._temp_directory = None
        self._delay_seconds = 30
        self._fixed_number_keyframes = None
        self._fixed_cadence_keyframes = 2
        self.time_multiplier_for_long_display = None
        self.local_directory = None
        self.all_wavelengths = ['0171', '0193', '0211', '0304', '0131', '0335', '0094']
        self.use_wavelengths = ['0171', '0193', '0211', '0304', '0131', '0335', '0094']
        self._resolution = 4096
        self.web_image_frame = None
        self.web_image_location = None
        self.web_paths = None
        self.file_ending = None
        self.run_time_offset = None
        self.time_file = None
        self.index_file = None
        self.debug_mode = False
        self.did_print = False
        self.Force_init = False
        self.list_of_default_hdus = ['t_integrated', 'original', 0, 1]
        self.original_image=None
        self.modified_image=None
        self.hdu_name = None
        self.start_time = time()
        self.is_first_run = True
        self._do_HMI = True
        self._mode = 'all'
        self._do_mirror = False
        
        # Movie Defaults
        self._download_files = False
        self._write_video = False
        self._delete = True
        self._make_compressed = False
        self._remove_old_images = False
        self._sonify_images = False
        self._sonify_limit = False
        self._do_171 = False
        self._do_304 = False
        self._do_one = False
        self._something_changed = False
        self._allow_muxing = True
        
        self._stop_after_one = False
        
        self._time_period = None
        self._range_in_days = 4
        self._cadence = 10 * u.minute
        self._exposure_time = 60  # seconds
        
        self._frames_per_second = 30
        self._bpm = 70
        self._debug_delay = 2
        
        self._run_type = "web"
        self.rez = None
        self.rbg_image = None
        self.center = None
        # self.changed = None
        
        # TODO remove this from params or something
        self._archive_url = None
        self._download_path = None
        self._time_path = None
        self._local_imgs_paths = None
        self._local_fits_paths = []
        self._do_multishot = True
        self._do_recent = True
        self._use_default_directories = True
        self.do_orig = True
        self.do_cat = False
        self.do_single = False
        
        self._fetchers = [LocalFetcher, ValidationProcessor]
        self._processors = []
        self._putters = [NullPutter]
        
        self._fet_rp = [None, None]
        self._proc_rp = []
        self._put_rp = [None]
        # self.set_default_values()
    
    # TODO: extract getter/setter logic
    # Main Functions
    
    def fetchers(self, _fetchers=None, rp=None):
        if _fetchers is not None:
            if type(_fetchers) not in [list]:
                self._fetchers = [_fetchers]
                self._fet_rp = [rp]
            else:
                self._fetchers.extend(_fetchers)
                self._fet_rp.extend([rp])
                
            if ValidationProcessor in self._fetchers:
                self._fetchers.remove(ValidationProcessor)
            self._fetchers.append(ValidationProcessor)
        
        return self._fetchers
    
    def processors(self, _processors=None, rp=None):
        if _processors is not None:
            if type(_processors) not in [list]:
                self._processors = [_processors]
                self._proc_rp = [rp]
            else:
                self._processors.extend(_processors)
                self._proc_rp.extend([rp])
        return self._processors
    
    def putters(self, _putters=None, rp=None):
        if _putters is not None:
            if type(_putters) not in [list]:
                self._putters =  [_putters]
                self._put_rp = [rp]
            else:
                self._putters.extend(_putters)
                self._put_rp.extend([rp])
        
        return self._putters
    
    ## Other
    
    # Directories
    
    def use_image_path(self, _image=None):
        if _image is not None:
            self._image = _image
        return self._image
    
    
    def base_directory(self, _base_directory=None):
        if _base_directory is not None:
            self._base_directory = _base_directory
        
        return self._base_directory
    
    def batch_name(self, _batch_name=None):
        if _batch_name is not None:
            self._batch_name = _batch_name
        return self._batch_name
    
    def archive_url(self, _archive_url=None):
        if _archive_url is not None:
            self._archive_url = _archive_url
        return self._archive_url
    
    def imgs_top_directory(self, _imgs_top_directory=None, make=False):
        if _imgs_top_directory is not None:
            self._imgs_top_directory = _imgs_top_directory
            if make:
                makedirs(self._imgs_top_directory, exist_ok=True)
        return self._imgs_top_directory
    
    def mods_directory(self, _imgs_directory=None, make=False):
        if _imgs_directory is not None:
            self._imgs_directory = _imgs_directory
            if make:
                makedirs(self._imgs_directory, exist_ok=True)
        return self._imgs_directory
    
    def fits_directory(self, _fits_directory=None, make=False):
        if _fits_directory is not None:
            self._fits_directory = _fits_directory
        if make and self._fits_directory is not None:
            makedirs(self._fits_directory, exist_ok=True)
        return self._fits_directory
    
    def temp_directory(self, _temp_directory=None, make=False):
        if _temp_directory is not None:
            self._temp_directory = _temp_directory
        if make:
            makedirs(self._temp_directory, exist_ok=True)
        return self._temp_directory
    
    def shortcut_directory(self, _shortcut_directory=None, make=False):
        if _shortcut_directory is not None:
            self._shortcut_directory = _shortcut_directory
        if make:
            makedirs(self._shortcut_directory, exist_ok=True)
        return self._shortcut_directory
    
    def movs_directory(self, _movs_directory=None, make=False):
        if _movs_directory is not None:
            self._movs_directory = _movs_directory
        if make:
            makedirs(self._movs_directory, exist_ok=True)
        return self._movs_directory
    
    
    
    
    def time_path(self, _time_path=None):
        if _time_path is not None:
            self._time_path = _time_path
        return self._time_path
    
    def curve_path(self, _curve_path=None):
        if _curve_path is not None:
            self._curve_path = _curve_path
        return self._curve_path
    
    def params_path(self, _params_path=None):
        if _params_path is not None:
            self._params_path = _params_path
        return self._params_path
    
    def local_fits_paths(self, _local_fits_paths=None):
        if _local_fits_paths is not None:
            self._local_fits_paths = _local_fits_paths
        return self._local_fits_paths
    
    def local_imgs_paths(self, _local_imgs_paths=None):
        if _local_imgs_paths is not None:
            self._local_imgs_paths = _local_imgs_paths
        return self._local_imgs_paths
    
    # BOOLEANS
    
    def fixed_cadence_keyframes(self, cadence=None):
        if cadence is not None:
            self._fixed_cadence_keyframes = cadence
            self._fixed_number_keyframes = None
        return self._fixed_cadence_keyframes
    
    def fixed_number_keyframes(self, number=None):
        if number is not None:
            self._fixed_cadence_keyframes = None
            self._fixed_number_keyframes = number
        return self._fixed_number_keyframes
    
    def run_type(self, _type=None):
        if _type is not None:
            self._run_type = _type
        return self._run_type
    
    def do_one(self, which=None, stop=False):
        if which is not None:
            self._do_one = which
            self.current_wave(which)
            # self.batch_name(which)
            self.stop_after_one(stop)
        
        return self._do_one
    
    def download_files(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._download_files = boolean
        # if self._download_files:
        #     self.something_changed(True)
        return self._download_files or self.local_fits_paths() in [None, []]
    
    def something_changed(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._something_changed = boolean
        return self._something_changed
    
    def overwrite_pngs(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._overwrite_pngs = boolean
        # if self._overwrite_pngs:
        #     self.something_changed(True)
        return self._overwrite_pngs or self.local_imgs_paths() in [None, []]
    
    def remake_norm_curves(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._remake_norm_curves = boolean
        return self._remake_norm_curves
    
    def write_video(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._write_video = boolean
        return self._write_video
    
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
    
    def use_default_directories(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._use_default_directories = boolean
        return self._use_default_directories
    
    def stop_after_one(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._stop_after_one = boolean
        return self._stop_after_one
    
    #
    # def stop_after_one(self, boolean=None):
    #     if boolean is not None:
    #         assert type(boolean) in [bool]
    #         self._stop_after_one = boolean
    #     return self._stop_after_one
    #
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
    
    def do_multishot(self, _do_multishot=None):
        if _do_multishot is not None:
            assert type(_do_multishot) in [bool]
            self._do_multishot = _do_multishot
        return self._do_multishot
    
    def cadence_minutes(self, cad=None):
        if cad is not None:
            self._cadence = cad * u.minute
        return self._cadence
    
    def exposure_time_seconds(self, _exposure_time=None):
        if _exposure_time is not None:
            self._exposure_time = _exposure_time
        return self._exposure_time
    
    def time_period(self, period=None):
        if period is not None:
            self._time_period = period
            self.tstart = period[0]
            self.tend = period[1]
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
    
    def set_waves_to_do(self, waves=None):
        
        if waves is not None:
            self.waves_to_do = [waves]
        elif self.do_one():
            self.waves_to_do = [self.do_one()]
        else:
            self.waves_to_do = self.all_wavelengths
        return self.waves_to_do
    
    def set_current_wave(self, wave=None):
        """Set the current wave parameter correctly"""
        
        if self.do_one():
            self.current_wave(self.do_one())
        else:
            self.current_wave(wave)
            
        self.set_current_wave_paths()
        self.make_directories()
        

    def get_wave_directory(self):
        """Define the root folder"""
        last = ''
        if type(self.current_wave()) is str:
            last = self.current_wave()
        base_directory = join(self.find_root_directory(), self.batch_name(), last)
        return self.base_directory(base_directory)
    
    def find_root_directory(self, root_directory_name = "sunback_images"):
        """Determine where to store the images"""
        
        # self.currently_local = False
        if self.currently_local: # True when run locally, False when run in panHelio
            self.root_directory = abspath(join("D://", root_directory_name))
        else:
            self.root_directory = abspath(root_directory_name)

        makedirs(self.root_directory, exist_ok=True)
        return self.root_directory
        
        #  Get the current path
        if __file__ in globals():
            this_file_path = dirname(abspath(__file__))
        else:
            this_file_path = abspath(getcwd())
        
        #  Escape Dropbox
        while "dropbox".casefold() in this_file_path.casefold():
            this_file_path = abspath(join(this_file_path, ".."))
        
        #  Name and create the root directory
        root_directory = join(this_file_path, root_directory_name)
        if not isdir(root_directory):
            makedirs(root_directory)
        self.root_directory = root_directory
        return self.root_directory
    
    def set_current_wave_paths(self):
        """Make the paths for current_wave"""
        # Define and Set Directories
        # print("Target: {}".format(self.current_wave))
        
        ## \\>Batch<\\>Wavelength<\\
        self.base_directory(abspath(self.get_wave_directory()))
        
        # Top Folders
        self.shortcut_directory(    abspath(join(self.base_directory(), '..', 'MOVS')))
        self.time_path(             abspath(join(self.base_directory(), "image_times.txt")))
        self.imgs_top_directory(    abspath(join(self.base_directory(), 'imgs')))
        self.movs_directory(        abspath(join(self.base_directory(), 'video')))
        self.analysis_directory =   abspath(join(self.base_directory(), "analysis"))

        # Fits Folders
        self.fits_directory(        abspath(join(self.imgs_top_directory(), 'fits')))
        self.temp_directory(        abspath(join(self.fits_directory(), "temp")))
        
        # Png Folders
        self.mods_directory(        abspath(join(self.imgs_top_directory(), 'png', 'mod')))
        self.orig_directory =       abspath(join(self.imgs_top_directory(), 'png', "orig"))
        self.cat_directory =        abspath(join(self.imgs_top_directory(), 'png', "cat"))
        
        # Analysis Folders
        
        norm_curves_name =      self.norm_curves_name or '{}_curves.txt'.format(self.current_wave())
        self.curve_path(        abspath(join(self.analysis_directory, norm_curves_name)))
        
        param_file_name =       '{}_params.txt'.format(self.current_wave())
        self.params_path(       abspath(join(self.analysis_directory, param_file_name)))
        
        
        
        # self.radial_hist_path = abspath(join(self.analysis_directory, param_file_name))
        
    def make_directories(self):
        # Make Directories
        makedirs(self.analysis_directory,   exist_ok=True)
        makedirs(self.imgs_top_directory(), exist_ok=True)
        makedirs(self.fits_directory(),     exist_ok=True)
        makedirs(self.orig_directory,       exist_ok=True)
        makedirs(self.mods_directory(),     exist_ok=True)
        makedirs(self.movs_directory(),     exist_ok=True)
        makedirs(self.cat_directory,        exist_ok=True)
        # Save Parameters
        self.save_to_txt()
        
    def make_file_paths(self, image_data):
        _, self.fits_save_path, _, _ = image_data
        fits_name = os.path.basename(self.fits_save_path)
        png_name =  fits_name.replace('fits', 'png')
        self.mod_path = join(self.mods_directory(), png_name)
        self.cat_path = self.mod_path.replace("mod", "cat")
        self.orig_path = self.mod_path.replace("mod", "orig")
        return self.mod_path, self.cat_path, self.orig_path


    def get_pre_radial_fig_paths(self):
        
        file_basename = self.file_basename or os.path.basename(self.use_image_path(self.image_data[1]))
        file_name = file_basename[:-5]

        bs = self.analysis_directory
        folder_name = "radial_hist_full"
        file_name_1 = 'full_{}.png'.format(file_name)
        save_path_1 = join(bs, folder_name, file_name_1)
        
        file_name_2 = 'zoom\\full_zoom_{}.png'.format(file_name)
        save_path_2 = join(bs, folder_name, file_name_2)
        
        makedirs(dirname(save_path_1), exist_ok=True)
        makedirs(dirname(save_path_2), exist_ok=True)
        
        return save_path_1, save_path_2
        



    
    def current_wave(self, _current_wave=None):
        if _current_wave is not None:
            self._current_wave = _current_wave
            if not self._current_wave:
                self._current_wave ='rainbow'
        return self._current_wave
    
    def check_real_number(self, number):
        assert type(number) in [float, int]
        assert number > 0
    
    def save_to_txt(self):  # , current_wave=None):
        try:
            # print("Txt Save Fail")
            # pass
            # Save contents of environment to text file
            # name = self.current_wave(current_wave)
            
            infoEnv = self
            with open(self.params_path(), 'w') as output:
                output.write(asctime() + '\n\n')
                myVars = (infoEnv.__class__.__dict__, vars(infoEnv))
                for pile in myVars:
                    for ii in sorted(pile.keys()):
                        if not callable(pile[ii]):
                            string = str(ii) + " : " + str(pile[ii]) + '\n'
                            output.write(string)
                    output.write('\n\n')
        except:
            print("Failed to print")
    
    
    def load_preset_time_settings(self, selection=None):
        """Load one of a few presets for the time settings"""
        if selection is not None:
            self.selection = selection.casefold()
        
        if self.selection in ['false', 'f', "False", None, False]:
            return False
        
        key_fixed_cadence = 1
        key_fixed_number = None
        switch =self.selection.casefold()
        # print("Loading {} cadence.".format(self.selection))
        if switch in ['slow', 's', 1, "1"]:
            cadence_minutes = 10 # One Forty Four Frames Per Day
            exposure_time_secs = 180 # Fifteen Frames per Frame
            self.selection = 'slow'
        
        elif switch in ['medium', 'm', 2, "2"]:
            cadence_minutes = 20 # Seventy Two Frames Per Day
            exposure_time_secs = 120  # Ten Frames per Frame
            self.selection = 'medium'
        
        elif switch in ['quick', 'q', 3, "3"]:
            cadence_minutes = 60 # Twenty Four Frames Per Day
            exposure_time_secs = 60  # Five Frames per Frame
            self.selection = 'quick'
        
        elif switch in ['ludacris', "ludicrous ", 'l', 4, "4"]:
            cadence_minutes = 3 * 60 # Eight Frames Per Day
            exposure_time_secs = 36  # Three Frames per Frame
            self.selection = 'ludacris'
        
        elif switch in ['ludacris', "ludicrous ", 'l2', 4, "4"]:
            cadence_minutes = 60 # 24 Frames Per Day
            exposure_time_secs = 36  # Three Frames per Frame
            self.selection = 'ludacris'
        
        elif switch in ['plaid', 'p', 5, "5"]:
            cadence_minutes = 6 * 60  # Four Frames per Day
            exposure_time_secs = 36  # Three Frames per Frame
            self.selection = 'plaid'
        
        else:
            return False
        
        if not self.did_print:
            print("Settings: {}".format(self.selection),
                  "\n  Cadence = {} Minutes ({} hours), [{}] per day".format(cadence_minutes, cadence_minutes/60, 24*60/cadence_minutes),
                  "\n  Exposure = {} Seconds".format(exposure_time_secs),)
            self.did_print = True
        
        # Set the Parameters
        # self.time_period(period=[tstart, tend])
        self.cadence_minutes(cadence_minutes)
        self.exposure_time_seconds(exposure_time_secs)
        self.fixed_cadence_keyframes(key_fixed_cadence)
        self.fixed_number_keyframes(key_fixed_number)
        
        return True
    
    def compare_fits_frames(self, compare_two_files=False):
        # path1 = "aia_lev1_171a_2014_11_04t03_50_11_34z_image_lev1.fits"
        # path2 = "aia_lev1_171a_2014_11_04t00_20_11_34z_image_lev1.fits"
        
        path1 = "aia_lev1_193a_2014_11_04t00_00_06_84z_image_lev1.fits"
        path2 = "aia_lev1_193a_2014_11_04t00_20_06_84z_image_lev1.fits"
        
        self_proc = Processor(self, quick=True)
        full_path1 = join(self.fits_directory(), path1)
        full_path2 = join(self.fits_directory(), path2)
        
        frame, wave, t_rec, center, int_time = Processor.load_last_fits_field(self_proc, full_path1)
        if compare_two_files:
            frame2, wave2, t_rec2, center2, int_time2 = Processor.load_last_fits_field(self_proc, full_path2)
        else:
            frame2, wave2, t_rec2, center2, int_time2 = Processor.load_first_fits_field(self_proc, full_path1)
            
            # Modifying
        frame3 = abs(frame2 - frame)
        
        #  Plotting
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, True, True)
        
        #  Make Color Scale the Same for All Frames
        mx1 = frame.flatten()[frame.argmax()]
        mx2 = frame2.flatten()[frame2.argmax()]
        mx3 = frame3.flatten()[frame3.argmax()]
        
        me1 = np.mean(frame)
        me2 = np.mean(frame2)
        me3 = np.mean(frame3)
        
        mn1 = frame.flatten()[frame.argmin()]
        mn2 = frame2.flatten()[frame2.argmin()]
        mn3 = frame3.flatten()[frame3.argmin()]
        
        stat_string = "\nMin, Mean, Max = {:0.2f}, {:0.2f}, {:0.2f}"
        stat_string1 = stat_string.format(mn1, me1, mx1)
        stat_string2 = stat_string.format(mn2, me2, mx2)
        stat_string3 = stat_string.format(mn3, me3, mx3)
        
        allmax = max(mx1, mx2)
        allmin = min(mn1, mn2)
        
        # Plot Commands
        ax1.imshow(frame, vmin=allmin, vmax=allmax)
        ax1.set_title(str(path1[14:30]) + stat_string1)
        
        ax2.imshow(frame2, vmin=allmin, vmax=allmax)
        ax2.set_title(str(path2[14:30]) + stat_string2)
        
        ax3.imshow(frame3, vmin=allmin, vmax=allmax)
        ax3.set_title("Diff" + stat_string3)
        
        #  Plot Formatting
        # plt.subplots_adjust(top=0.987,
        #                     bottom=0.025,
        #                     left=0.023,
        #                     right=0.977,
        #                     hspace=0.053,
        #                     wspace=0.2)
        fig.set_size_inches(5, 12)
        plt.tight_layout()
        plt.show(block=True)
    
    def reprocess_mode(self, _reprocess_mode=None):
        """Pick how it should handle frames that already exist
            options are:
                skip    - do nothing
                redo    - pull from prev out_array to recompute same as last time
                add     - pull from prev out_array to recompute but store seperately
                reset   - pull from first out_array to recompute from scratch
                double  - pull from current out_array to double the filter
        
        """
        if _reprocess_mode is not None:
            self._reprocess_mode = _reprocess_mode
        return self._reprocess_mode
    
    def delete_old(self, _delete=None):
        if _delete is not None:
            self._delete = _delete
        return self._delete
    
    def delay_seconds(self, _delay=None):
        if self.is_debug():
            self._delay_seconds = self._debug_delay
        elif _delay is not None:
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
            self.local_directory = self.find_root_directory()
        
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
        # if 'temp' in current_wave:
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
            # print('Done')
    
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
        
        # self.movie_folder = abspath(join(base_directory, "movies\\"))
        # self.video_name_stem = join(self.movie_folder, '{}_{}_movie{}'.format(current_wave, strftime('%m%d_%H%M'), '{}'))
        
        # makedirs(self.base_directory(), exist_ok=True)
        # # SunbackMovie Parameters
        #
        # # Sunback Still Parameters
        # #  Set Delay Time for Background Rotation
        # self.delay_seconds(30 * self.seconds)
        # self.set_time_multiplier(3)
        #
        # # Set File Paths
        # self.set_local_directory()
        # self.time_file = join(self.local_directory, 'time.txt')
        # self.index_file = join(self.local_directory, 'index.txt')
        #
        # # Set Wavelengths
        # self.set_wavelengths(['0171', '0193', '0211', '0304', '0131', '0335', '0094'])
        #
        # # Set Resolution
        # self.set_download_resolution(2048)
        #
        # # Set Web Location
        # self.set_web_image_frame("https://sdo.gsfc.nasa.gov/assets/img/latest/latest_{}_{}")
        #
        # # # Add extra images
        # # new_web_path_1 = "https://sdo.gsfc.nasa.gov/assets/img/latest/f_211_193_171pfss_{}.jpg".format(self.resolution)
        # # self.append_to_web_paths(new_web_path_1, 'PFSS')
        #
        # # Select File Ending
        # self.set_file_ending("{}_Now.png")
        #
        # return 0