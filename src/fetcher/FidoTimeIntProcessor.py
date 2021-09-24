import os
import shutil
from copy import copy

from os.path import join, basename
from time import strptime, mktime
import sys
from parfive import Downloader
from sunpy.net import Fido, attrs
import numpy as np
import astropy.units as u

from fetcher.FidoFetcher import FidoFetcher
jsoc_email = "chris.gilly@colorado.edu"
_verb = False
import datetime

default_base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


def vprint(in_string, verb=_verb, *args, **kwargs):
    if verb:
        print(in_string, *args, **kwargs)

class FidoTimeIntProcessor(FidoFetcher):
    name = filt_name = "Time Integration"
    in_name = -1
    out_name = "t_integrated"
    description = "Get many frames around the keyframe and sum them"
    progress_verb = 'Binning'
    finished_verb = "Summed"
    dopng = False
    temp_folder = ''
    exposure_paths = []
    
    ## Structure ###
    
    def __init__(self, params=None, quick=False, rp=False):
        # Initialize class variables
        super().__init__(params, quick, rp)
        self.orig_t_int = None
        self.keyframe_fits_path = None
        self.main_time_period = None
        self.subname = None
        # self.fetch()

    # Main Call
    def fetch(self, params=None, quick=False, rp=None, verb=None):
        """ Get the fits files from Fido """
        if verb is not None:
            self.verb = verb
        self.fido_get_fits_short_cadence()

    def fido_get_fits_short_cadence(self):
        self.load(self.params, wave=self.params.current_wave())
        vprint(" v Fetching Fits Files: {}".format(self.params.current_wave()), self.verb)
        if self.get_files():
            self.print_load_banner(verb=self.verb)
            self.prep_temp_folder()
            self.download_fits_series()
            self.validate_download()
        else:
            vprint(" ^ Using {} Cached Fits Files\n".format(self.params.n_fits), self.verb)

    
    def get_files(self):
        return self.params.download_files() or self.reprocess_mode() or not self.verb

    def setup(self):
        os.makedirs(self.params.temp_directory(), exist_ok=True)
    
    def cleanup(self):
        self.reset_params()
        self.sum_subframes()
        
        # self.delete_temp() # TEMPCHANGE

    def do_fits_function(self, fits_path, in_name=None):
        """This is the thing that will be executed on every file
            In this case, that thing is time integration
        """
        if fits_path is None:
            return False
        
        if self.do_exposure(fits_path):
            # Get the Images
            self.gather_subframes(fits_path, in_name)
        
            # Sum them
            return self.changed
        
        return None
    
    def do_exposure(self, fits_path):
        """Do we need to do time integration here?"""
        self.keyframe_fits_path = fits_path
        in_name = self.set_hdul_in_name(fits_path)
        need_exposure = self.params.exposure_time_seconds() > 0
        have_input = in_name is not None
        already_made = self.out_name in self.hdu_name_list
        reprocess = self.reprocess_mode()
        do_exposure = need_exposure and have_input and (not already_made or reprocess)
        return do_exposure

    def download_fits_series(self):
        self.define_range()
        self.fido_check_for_fits()
        if self.fido_search_found:
            self.prep_temp_folder()
            self.fido_parse_result()
            self.fido_download_fits_ensured(temp=True, hold=True)
        else:
            print("\n     No Images Found\n")
    
    def gather_subframes(self, fits_path, in_name):
        # Parse the Keyframe Time
        self.init_integration_period(fits_path, in_name)
        
        # Search fido for those frames + Download the Files
        self.fetch(self.params, True, verb=False)
        
    def init_integration_period(self, fits_path, in_name):
        
        
        self.subname = fits_path.split('\\')[-1][:-5]
        # self.subname = basename(fits_path.split('.')[0])
        
        keyframe, wave, t_rec, center, t_int = self.load_a_fits_field(fits_path, in_name)
        self.orig_t_int = t_int
        self.original = keyframe
        self.changed = self.original / self.orig_t_int

        # Define new exposure time window
        self.main_time_period = self.params.time_period([self.params.tstart, self.params.tend])
        self.set_time_range_duration(t_start=t_rec, duration_seconds=self.params.exposure_time_seconds())
        self.params.do_recent(False)
        self.params.cadence_minutes(10. / 60.)
        
    def reset_params(self):
        # Reset the main time period
        self.params.time_period(self.main_time_period)
        self.params.load_preset_time_settings()
        self.define_range()
        
    def get_exposure_paths(self):
        exposure_files = os.listdir(self.temp_folder)
        self.exposure_paths = [join(self.temp_folder, path) for path in exposure_files]
        return self.exposure_paths
        
    def sum_subframes(self):
        # frame_array = self.original + 0
        self.get_exposure_paths()
        self.int_tm_tot = 0
        for ii, path in enumerate(self.exposure_paths):
            try:
                if not os.path.isdir(path):
                    frame, wave, t_rec, center, int_time = self.load_last_fits_field(path)
                    self.changed += (frame / int_time)
                    self.int_tm_tot += int_time
                # self.force_delete(path)
            except PermissionError as e:
                print("Sum Subframes:: ", e)
            except TypeError as e:
                print("Sum Subframes:: ", e)
                
        self.changed *= self.orig_t_int
        self.changed = np.asarray(self.changed, dtype=np.float32)
        # self.delete_temp_folder()
        
        

        
    ## TEMP FOLDER IO ##
    def prep_temp_folder(self):
        self.params.download_files(True)
        # self.temp_folder = self.params.temp_directory()
        self.temp_folder = join(self.params.temp_directory(), self.subname)
        os.makedirs(self.temp_folder, exist_ok=True)
        # self.delete_temp_folder_items()
        
    def delete_temp(self, delete_folder_too=True):
        if delete_folder_too:
            self.delete_temp_folder()
        else:
            self.delete_temp_folder_items()
        
    def delete_temp_folder(self):
        if os.path.isdir(self.temp_folder):
            shutil.rmtree(self.temp_folder)
            
    def delete_temp_folder_items(self):
        for root, dirs, files in os.walk(self.temp_folder):
            for file in files:
                self.force_delete(file, root)

    @staticmethod
    def force_delete(file, root=''):
        if not os.path.isdir(file):
            os.remove(os.path.join(root, file))
        else:
            shutil.rmtree(file)
        
        
    ## Time Range ##
    def set_time_range_duration(self, t_start, duration_seconds=60):
        
        # Get a start_timestamp datetime
        t_start_struct = strptime(t_start[:-4], "%Y-%m-%dT%H:%M:%S")
        t_start_dt = datetime.datetime.fromtimestamp(mktime(t_start_struct))
        
        # Do math
        delta = datetime.timedelta(seconds=duration_seconds)
        t_end_dt = t_start_dt + delta
        
        # Get the formatted outputs
        t_start_out = t_start_dt.strftime('%Y/%m/%d %H:%M:%S')
        t_end_out = t_end_dt.strftime('%Y/%m/%d %H:%M:%S')
        
        # Set to parameters object
        self.params.time_period(period=[t_start_out, t_end_out])
        self.define_range()
        return [t_start_out, t_end_out]
        # self.params.do_multishot()

    

        
    # def remove_and_mark_redownload(self, filename):
    #     fitsPath = join(self.fits_folder, filename[:-5] + '.fits')
    #     self.redownload.append(filename)
    #     os.remove(fitsPath)
    #
    # def remove_fits_and_png(self, filename):
    #     fitsPath = join(self.fits_folder, filename[:-5] + '.fits')
    #     pngPath = join(self.image_folder, filename[:-5] + '.png')
    #     try:
    #         os.remove(fitsPath)
    #     except PermissionError as e:
    #         print(e)
    #     try:
    #         os.remove(pngPath)
    #     except FileNotFoundError as e:
    #         # print(e)
    #         pass
    #
    # def fido_download_fits_ensured(self):
    #     overwrite = True
    #     print(" *     Downloading...")
    #     results = Fido.fetch(self.fido_search_result, path=self.params.temp_directory(),
    #                          downloader=Downloader(progress=True, file_progress=False, max_conn=100,
    #                                                overwrite=overwrite))
    #     n_fits = len(self.exposure_paths)
    #     if n_fits:
    #         print(" ^     Successfully Downloaded {} Files\n".format(n_fits), flush=True)
    #     else:
    #         print(" ^     Unable to Download...Try again Later.")
    #         raise(FileNotFoundError(" Unable to Download...Try again Later."))
    #     sys.stdout.flush()
    #     return results
    #
    
    # def download_fits_series(self):
    #     self.fido_check_for_fits()
    #     if self.fido_search_found:
    #         self.fido_parse_result()
    #         self.fido_download_fits_ensured()
    #     else:
    #         print("\n     No Images Found\n")
    
    
    # def define_range(self):
    #
    #
    # @staticmethod
    # def define_duration_range(start_timestamp, duration): ## THIS IS NOT IMPLEMENTED, and put it back where you got it
    #     """Given a short and a long cadence, make an input to fido that gets that"""
    #     start_struct = datetime.datetime.strptime(start_timestamp, '%Y/%m/%d %H:%M:%S')
    #     end_struct = datetime.datetime.strptime(start_timestamp + duration, '%Y/%m/%d %H:%M:%S')
    #     return get_time_lists(start_struct, end_struct) #something that makes fido do the right thing by itself
    #
    #