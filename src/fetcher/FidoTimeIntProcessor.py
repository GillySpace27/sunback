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


import datetime

default_base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images

global global_verb
global_verb=False

def vprint(in_string, verb=None, *args, **kwargs):
    global global_verb
    if verb is not None:
        global_verb = verb
    if FidoTimeIntProcessor.verb or global_verb:
        print(in_string, *args, **kwargs)

class FidoTimeIntProcessor(FidoFetcher):
    name = filt_name = "Time Integration {} Seconds"
    in_name = -1
    out_name = "t_integrated"
    description = "Get many frames around the keyframe and sum them"
    finished_verb = "Summed"
    dopng = False
    temp_folder = ''
    exposure_paths = []
    
    ## Structure ###
    
    def __init__(self, params=None, quick=False, rp=False):
        # Initialize class variables
        super().__init__(params, quick, rp)
        self. progress_verb = 'Time Integrating: {} Seconds'.format(self.params.exposure_time_seconds)
        self.name = self.name.format(self.params.exposure_time_seconds)
        self.do_delete = True
        self.orig_t_int = None
        self.keyframe_fits_path = None
        self.main_time_period = None
        self.subname = 'default'
        self.hold = False
        self.verb=False
        # self.fetch()

    def should_get_files(self):
        return self.params.download_files() or self.reprocess_mode() or not self.verb

    def setup(self):
        os.makedirs(self.params.temp_directory(), exist_ok=True)
        self.params.do_temp = True

    def download_fits_series(self, temp=True, hold=None):
        if hold is None:
            hold = False # TODO Fix this
        self.define_range()
        self.fido_check_for_fits()
        if self.fido_search_found_num:
            self.fido_parse_result()
            self.fido_download_fits_ensured(temp=temp, hold=hold)
        else:
            print("\n     No Images Found\n")
    
    def cleanup(self):
        self.reset_params()
        
        if self.hold:
            self.fido_download_fits_ensured(temp=True)
            self.sum_subframes()
        
        self.delete_temp()  # TEMPCHANGE

    def do_fits_function(self, fits_path, in_name=None):
        """This is the thing that will be executed on every file
            In this case, that thing is time integration
        """
        if fits_path is None:
            return False
        self.fits_path = fits_path
        
        self.in_name = self.set_hdul_in_name(fits_path)
        
        if self.should_do_exposure(fits_path):
            # Get the Images
            self.gather_subframes(fits_path)
            # Sum them
            if not self.hold:
                self.sum_subframes()
                if self.do_delete:
                    self.delete_temp_folder_items()
            return self.params.modified_image
        
        return None
    
    def should_do_exposure(self, fits_path):
        """Do we need to do time integration here?"""
        self.keyframe_fits_path = fits_path
        in_name = "original_image" # self.set_hdul_in_name(fits_path)
        need_exposure = self.params.exposure_time_seconds() > 0
        have_input = in_name is not None
        already_made = self.out_name in self.hdu_name_list
        reprocess = self.reprocess_mode()
        do_exposure = need_exposure and have_input and (not already_made or reprocess)
        return do_exposure
 
    def gather_subframes(self, fits_path):
        # Parse the Keyframe Time
        self.init_integration_period(fits_path)
        
        # Search fido for those frames + Download the Files
        self.fetch(self.params, quick=True, verb=False)
        
    def init_integration_period(self, fits_path):
        
        # in_name = self.in_name
        self.subname = fits_path.split('\\')[-1][:-5]
        # self.subname = basename(fits_path.split('.')[0])
        self.set_hdul_in_name(fits_path)
        
        keyframe, wave, t_rec, center, t_int = self.load_a_fits_field(fits_path, self.in_name)
        self.orig_t_int = t_int
        self.params.original_image = keyframe
        self.params.modified_image = np.zeros_like(keyframe, dtype=np.float32)

        # Define new exposure time window
        self.main_time_period = self.params.time_period([self.params.tstart, self.params.tend])
        self.set_time_range_duration(t_start=t_rec, duration_seconds=self.params.exposure_time_seconds())
        self.params.do_recent(False)
        self.params.cadence_minutes(10. / 60.)
        self.out_dtype = np.float32
        
        
    def reset_params(self):
        # Reset the main time period
        self.params.time_period(self.main_time_period)
        self.params.load_preset_time_settings()
        self.define_range()
        
    def get_exposure_paths(self):
        self.prep_temp_folder()
        exposure_files = os.listdir(self.temp_folder)
        self.exposure_paths = [join(self.temp_folder, path) for path in exposure_files]
        self.exposure_paths.append(self.fits_path)
        return self.exposure_paths
        
    def sum_subframes(self):
        # self.verb=False
        # vprint("Summing Arrays", False)
        self.get_exposure_paths()
        self.int_tm_tot = 0
        self.n_exposures = 0
        self.params.modified_image = np.zeros_like(self.params.modified_image, dtype=np.float32)
        for ii, path in enumerate(self.exposure_paths):
            # vprint(path, self.verb)
            try:
                if not os.path.isdir(path):
                    frame, wave, t_rec, center, int_time = self.load_a_fits_field(path, -1)
                    self.orig_t_int = self.orig_t_int or int_time
                    self.params.modified_image += (frame / int_time)
                    self.int_tm_tot += int_time
                    self.n_exposures += 1
                # self.force_delete(path, do=self.do_delete)
            except (PermissionError, TypeError, ValueError) as e:
                print("Sum Subframes:: ", e)
            # except TypeError as e:
            #     print("Sum Subframes:: ", e)
                
        self.params.modified_image /= self.n_exposures
        self.params.modified_image *= self.orig_t_int #TODO remove this line to make the curves be per second
        self.params.modified_image = np.asarray(self.params.modified_image, dtype=self.out_dtype)

    ## TEMP FOLDER IO ##
    def prep_temp_folder(self):
        self.params.download_files(True)
        self.temp_folder = self.params.temp_directory()
        # self.temp_folder = join(self.params.temp_directory(), self.subname)
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
            
    def delete_temp_folder_items(self, folder=None):
        directory = folder if folder is not None else self.temp_folder
        for root, dirs, files in os.walk(directory):
            for file in files:
                self.force_delete(file, root)

    @staticmethod
    def force_delete(file, root='', do=True):
        if do:
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
        delta = max(duration_seconds - 21, 1)
        duration = datetime.timedelta(seconds=delta)
        shift = datetime.timedelta(seconds=delta/1.5)
        t_end_dt   = t_start_dt + shift + duration
        t_start_dt = t_start_dt + shift
        
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
    #     if self.fido_search_found_num:
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