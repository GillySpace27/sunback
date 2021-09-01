import os
import shutil

from os import remove
from os.path import join, basename
from time import strptime, mktime
import sys
from parfive import Downloader
from sunpy.net import Fido, attrs
import numpy as np
from astropy.io import fits
from os import stat
from fetcher.Fetcher import Fetcher
from utils.time_util import parse_time_string_to_local
import astropy.units as u
_verb = True
import datetime
from utils.time_util import define_time_range, define_recent_range

default_base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


def vprint(in_string, verb=_verb, *args, **kwargs):
    if verb:
        print(in_string, *args, **kwargs)


class FidoFetcher(Fetcher):
    """Gets some data"""
    description = "Get Fits Files from the Internet using Fido"
    verb = _verb
    def __init__(self, params=None):
        if params is not None:
            super().__init__(params=params)
    
    ## Main Fetch Logic
    def fetch(self, params=None):
        self.__init__(params)
        
        """ Find the Most Recent Images """
        for current_wave in self.waves_to_do:
            self.fido_get_fits(current_wave)
    
    def fido_get_fits(self, current_wave):
        self.load(self.params, wave=current_wave)
        if self.params.download_images():
            vprint("\n  Fetching Fits Files: {}".format(self.current_wave), self.verb)
            self.download_fits_series()
            self.validate_download()
        else:
            print("   Using {} Cached Fits Files\n".format(self.params.n_fits))
    
    def download_fits_series(self):
        self.define_range()
        self.fido_check_for_fits()
        if self.fido_num:
            self.fido_parse_result()
            self.fido_download_fits()
        else:
            print("\n     No Images Found\n")
    
    def fido_check_for_fits(self):
        """Find the science images"""
        vprint("   Looking for Images of {} from {} to {}...".format(
                self.current_wave, self.start_string, self.end_time_string), flush=True, end='', verb=self.verb)
        
        # Search for records from the internet
        self.fido_result = Fido.search(attrs.Time(self.start_time, self.end_time), attrs.Instrument('aia'),
                                       attrs.Wavelength(int(self.current_wave) * u.angstrom),
                                       attrs.Sample(
                                               self.params.cadence_minutes()))  # , attrs.Resolution(self.params.resolution()))  # , a.vso.Provider('jsoc'))
        self.fido_num = self.fido_result.file_num
        # print("Found {}".format(self.fido_num))
    
    def fido_parse_result(self):
        """Examine the search results"""
        
        first_result = self.fido_result.get_response(0)[0]
        last_result = self.fido_result.get_response(0)[-1]
        
        self.name = first_result.wave.wavemin
        
        time_start = first_result.time.start
        time_end = last_result.time.end
        
        begin_time = parse_time_string_to_local(''.join(i for i in time_start if i.isdigit()), 2)[0]
        end_time = parse_time_string_to_local(''.join(i for i in time_end if i.isdigit()), 2)[0]
        
        self.extra_string = "from {} to {}".format(begin_time, end_time)
        
        self.startTime = time_start  # #self.parse_time_string_to_local(str(time_start.fits), 2)[0]
        self.endTime = time_end  # ''.join(i for i in time_end.iso if i.isdigit())   #self.parse_time_string_to_local(str(time_end.fits), 2)[0]
        
        vprint("\n     Search Found {} Images {}...".format(self.fido_num, self.extra_string), flush=True, verb=self.verb)
        
        while len(self.name) < 4:
            self.name = '0' + self.name
    
    def fido_download_fits(self):
        overwrite = False
        results = Fido.fetch(self.fido_result, path=self.params.fits_directory(),
                             downloader=Downloader(progress=True, file_progress=False, max_conn=100,
                                                   overwrite=overwrite))
        print("     Successfully Downloaded {} Files\n".format(len(results)), flush=True)
        sys.stdout.flush()
        return results
    
    # Validation
    def validate_download(self):
        # try:
        #     self.set_output_paths()
        # except:
        #     set_output_paths(self)
        # self.list_requested_files()
        # self.local_fits_paths = list_files_in_directory(self.fits_folder)
        
        pass
        
        # if self.params.delete_old():
        #     self.remove_all_old_fits_pngs()
        #     self.remove_all_old_pngs()
        #
        # working = False
        # if working:
        #     self.validate_fits()
        #     self.redownload_bad_fits()
        #
        # self.fido_download_fits()
        
        # self.find_missing_images()
        # self.get_missing_images()
    
    def list_requested_files(self):
        self.requested_files = []
        self.requested_response = []
        for ii in np.arange(self.fido_num):
            self.requested_files.append(self.fido_result.get_response(0)[ii]['fileid'].casefold())
            self.requested_files.append(self.fido_result.get_response(0)[ii]['time']['start'])
    
    def remove_all_old_pngs(self):
        requested_pngs = [x.replace('fits', 'png') for x in self.local_fits_paths]
        png_directory = join(self.params.imgs_directory(), self.current_wave, 'png')
        got_png = self.params.local_imgs_paths()  # list_files_in_directory(png_directory, 'png')
        remove_count = 0
        for png_path in got_png:
            if png_path not in requested_pngs:
                try:
                    remove(join(png_directory, png_path))
                    remove_count += 1
                except FileNotFoundError as e:
                    # print(e)
                    pass
        if remove_count > 0:
            print("{} old pngs deleted".format(remove_count))
    
    def remove_all_old_fits_pngs(self):
        keep = []
        self.file_size = []
        for local_file in self.local_fits_paths:
            if local_file not in self.requested_files:
                start = self.parse_filename_to_time(local_file)
                if start not in self.requested_files:
                    self.remove_fits_and_png(local_file)
            else:
                keep.append(local_file)
                self.file_size.append(stat(join(self.fits_folder, local_file)).st_size)
            self.params.local_fits_paths(keep)
        
        if len(self.redownload) > 0:
            print("        Deleting old files...", end='')
            print("  Success! Deleted {} old images".format(len(self.redownload)))
    
    def parse_filename_to_time(self, local_file):
        try:
            ifirst = 13 if '94' in local_file else 14
            stub = local_file[ifirst:-20]
            fmt_A = '%Y_%m_%dt%H_%M_%S'
            fmt_B = '%Y%m%d%H%M%S'
            # return stub.replace(['_', 't'], '')
            return datetime.datetime.strptime(stub, fmt_A).strftime(fmt_B)
        except:
            stub = local_file[3:-10].replace('_', '')
            return stub
    
    def out_of_range(self, hdul):
        print('A')
        pass
        return False
    
    def remove_and_mark_redownload(self, filename):
        fitsPath = join(self.fits_folder, filename[:-5] + '.fits')
        self.redownload.append(filename)
        remove(fitsPath)
    
    def remove_fits_and_png(self, filename):
        fitsPath = join(self.fits_folder, filename[:-5] + '.fits')
        pngPath = join(self.image_folder, filename[:-5] + '.png')
        try:
            remove(fitsPath)
        except PermissionError as e:
            print(e)
        try:
            remove(pngPath)
        except FileNotFoundError as e:
            # print(e)
            pass
        
    def delete_temp_folder(self):
        shutil.rmtree(self.temp_folder)
    
    def validate_fits(self):
        from statistics import mode
        # self.file_size_mode = mode(self.file_size)
        self.redownload = []
        for local_file in self.params.local_fits_paths():
            abs_path = join(self.fits_folder, local_file)
            with fits.open(abs_path) as hdul:
                hdul.verify('silentfix+warn')
                delete = False
                try:
                    try:
                        hh = 0
                        total_counts = np.nansum(hdul[hh].data)
                    except Exception as e:
                        print(e)
                        hh = 1
                        delete = True
                        total_counts = np.nansum(hdul[hh].data)
                        delete = False
                    this_size = stat(abs_path).st_size
                    data = hdul[hh].data
                    if total_counts < 0:  # or not this_size == self.file_size_mode:
                        delete = True
                except TypeError as e:
                    print(e)
                    delete = True
            
            if delete:
                self.remove_and_mark_redownload(local_file)
        n_corrupt = len(self.redownload)
        if n_corrupt:
            print("        Deleted {} corrupted files. Re-downloading...".format(n_corrupt))
    
    def redownload_bad_fits(self):
        if len(self.redownload) > 0:
            self.redownload = []
            self.fido_get_fits()
    
    def define_range(self):
        """Defines the time range of imagery desired"""
        if self.params.do_recent():
            self.parse_time(*define_recent_range(self.params.range()))
        # elif self.params.exposure_time():
        #     self.parse_time(*define_duration_range(*self.params.time_period()))
        else:
            start, end = self.params.time_period()
            self.parse_time(*define_time_range(start, end))
    
    def parse_time(self, start, end):
        """Unpacks the time lists"""
        self.start_time, self.start_time_long, self.start_string = start
        self.end_time, self.end_time_long, self.end_time_string = end


class FidoMultiFrameProcessor(FidoFetcher):
    filt_name = "Temporal Binning"
    description = "Get many frames around the keyframe and sum them"
    dopng = False
    out_name = "t_integrated"
    temp_folder = ''
    
    def __init__(self, params=None):
        if params is not None:
            super().__init__(params=params)
            self.set_function(self.do_fits_function)
            self.exposure_paths = []
    
    def fido_download_fits(self):
        overwrite = False
        self.temp_folder = join(self.params.fits_directory(), "temp_" + self.subname)
        self.exposure_paths.extend(Fido.fetch(self.fido_result, path= self.temp_folder,
                                              downloader=Downloader(progress=verb, file_progress=False, max_conn=100,
                                                                    overwrite=overwrite)))
        vprint("     Successfully Downloaded {} Files\n".format(len(self.exposure_paths)), flush=True, verb=self.verb)
        sys.stdout.flush()
    
    def do_fits_function(self, fits_path, in_name):
        """This is the thing that will be executed on every file"""
        
        # Get the Images
        self.gather_subframes(fits_path, in_name)
        
        # Sum them
        self.sum_subframes()
        
        
        # Store array into keyframe
        
        
        # Delete exposure files
        self.delete_temp_folder()
        
        # pass frame
        return self.changed
    
    def gather_subframes(self, fits_path, in_name):
        # Parse the Keyframe Time
        keyframe, wave, t_rec, center = self.load_a_fits_field(fits_path, in_name)
        self.original = keyframe
        self.changed = None
        self.exposure_paths = [fits_path]
        self.subname = basename(fits_path.split('.')[0])
        
        # Define new exposure time window
        self.make_time_range_duration(t_start=t_rec, duration_seconds=self.params.exposure_time())
        self.params.do_recent(False)
        self.params.cadence_minutes(10. / 60.)
        
        # Search fido for those frames + Download the Files
        self.fetch(self.params)
    
    def sum_subframes(self):
        frame_array = np.zeros_like(self.original)
        for path in self.exposure_paths:
            frame_array += self.load_last_fits_field(path)[0]
        self.changed = frame_array
    

    def make_time_range_duration(self, t_start, duration_seconds=60):
        
        # Get a start datetime
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
    
    def download_fits_series(self):
        self.fido_check_for_fits()
        if self.fido_num:
            self.fido_parse_result()
            self.fido_download_fits()
        else:
            print("\n     No Images Found\n")
    
    def fido_check_for_fits(self):
        """Find the science images"""
        vprint("   Looking for Images of {} from {} to {}...".format(
                self.current_wave, self.start_string, self.end_time_string), flush=True, end='', verb=self.verb)
        
        # Search for records from the internet
        self.fido_result = Fido.search(attrs.Time(self.start_time, self.end_time), attrs.Instrument('aia'),
                                       attrs.Wavelength(int(self.current_wave) * u.angstrom),
                                       attrs.Sample(
                                           self.params.cadence_minutes()))  # , attrs.Resolution(self.params.resolution()))  # , a.vso.Provider('jsoc'))
        self.fido_num = self.fido_result.file_num
        # print("Found {}".format(self.fido_num))
        

        
        
        
        
        
    # def define_range(self):
    #
    #
    # @staticmethod
    # def define_duration_range(start, duration): ## THIS IS NOT IMPLEMENTED, and put it back where you got it
    #     """Given a short and a long cadence, make an input to fido that gets that"""
    #     start_struct = datetime.datetime.strptime(start, '%Y/%m/%d %H:%M:%S')
    #     end_struct = datetime.datetime.strptime(start + duration, '%Y/%m/%d %H:%M:%S')
    #     return get_time_lists(start_struct, end_struct) #something that makes fido do the right thing by itself
    #
    #