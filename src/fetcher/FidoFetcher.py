from calendar import timegm
from os import makedirs, listdir, remove
from os.path import join, abspath, exists, dirname
from time import time, timezone, localtime, struct_time, strftime, sleep
import sys
from parfive import Downloader
from sunpy.net import Fido, attrs
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from os import stat

from fetcher.Fetcher import Fetcher
# from movie.modifyMovie import Sonifier
from fetcher.LocalFitsFetcher import LocalFitsFetcher
from utils.file_util import discover_best_data_directory, list_files_in_directory, build_paths, set_output_paths
from utils.time_util import parse_time_string_to_local
import astropy.units as u

from sunpy.instr.aia import aiaprep

import datetime

# aia1 = sunpy.map.Map(file_download[0])
# aia = aiaprep(aia1)
from utils.time_util import define_time_range, define_recent_range


class FidoFetcher(Fetcher):
    """Gets some data"""
    wavelengths = ['0171', '0193', '0211', '0304', '0131', '0335', '0094']
    
    def __init__(self, params, base_url=None, base_dir_path=discover_best_data_directory()):
        super().__init__()  # Initializes class variables
        
        # Timer
        self.beginTime = time()
        
        # Parse Inputs
        self.load_params(params, base_url, base_dir_path)
    
    def load_params(self, params, base_url, base_dir_path):
        """Sets the parameter object and initializes other inputs"""
        self.params = params
        self.params.archive_url(base_url)
        self.params.img_directory(base_dir_path)
        
        self.set_wavelengths()
        self.define_range()
    
    def set_wavelengths(self):
        """Selects which wavelengths to use"""
        if self.params.do_one():
            self.waves_to_do = [self.params.do_one()]
        else:
            self.waves_to_do = self.wavelengths
    
    def define_range(self):
        """Defines the time range of imagery desired"""
        if self.params.do_recent():
            self.parse_time(*define_recent_range(self.params.range()))
        else:
            self.parse_time(*define_time_range(*self.params.time_period()))
    
    def parse_time(self, start, end):
        """Unpacks the time lists"""
        self.start_time, self.start_time_long, self.start_string = start
        self.end_time, self.end_time_long, self.end_time_string = end
    
    ## Main Runner
    def fetch(self):
        """ Find the Most Recent Images """
        for self.current_wave in self.waves_to_do:
            self.fido_get_fits()
        self.params.local_fits_paths(self.temp_fits_pathbox)


    ### Utils
    
    def fido_get_fits(self):
        build_paths(self)
        if self.params.download_images():
            self.download_fits_series()
            self.validate_download()
        # else:
            # LocalFitsFetcher(self.params, self.current_wave).fetch()
        set_output_paths(self)
        

    
    
    
    ## Work
    def download_fits_series(self):
        self.fido_check_for_fits()
        if self.fido_num:
            self.fido_parse_result()
            self.fido_download_fits()
    
    def fido_check_for_fits(self):
        """Find the science images"""
        print("\n>Looking for Science Images of {} from {} to {}...".format(
                self.current_wave, self.start_string, self.end_time_string), flush=True, end='')
        
        # Search for records from the internet
        self.fido_result = Fido.search(attrs.Time(self.start_time, self.end_time), attrs.Instrument('aia'),
                                       attrs.Wavelength(int(self.current_wave) * u.angstrom),
                                       attrs.Sample(
                                               self.params.cadence_minutes()))  # , attrs.Resolution(self.params.resolution()))  # , a.vso.Provider('jsoc'))
        self.fido_num = self.fido_result.file_num
        print("Found {}".format(self.fido_num))
    
    def fido_parse_result(self):
        """Examine the search results"""
        
        first_result = self.fido_result.get_response(0)[0]
        last_result = self.fido_result.get_response(0)[-2]
        
        self.name = first_result.wave.wavemin
        
        time_start = first_result.time.start
        time_end = last_result.time.end
        
        begin_time = parse_time_string_to_local(''.join(i for i in time_start if i.isdigit()), 2)[0]
        end_time = parse_time_string_to_local(''.join(i for i in time_end if i.isdigit()), 2)[0]
        
        self.extra_string = "from {} to {}".format(begin_time, end_time)
        
        self.startTime = time_start  # #self.parse_time_string_to_local(str(time_start.fits), 2)[0]
        self.endTime = time_end  # ''.join(i for i in time_end.iso if i.isdigit())   #self.parse_time_string_to_local(str(time_end.fits), 2)[0]
        
        print("           Search Found {} Images {}...".format(self.fido_num, self.extra_string), flush=True)
        
        while len(self.name) < 4:
            self.name = '0' + self.name
    
    def fido_download_fits(self):
        try:
            print("        Downloading {}...".format(self.fido_num), end='', flush=True)
            # err = sys.stderr
            # sys.stderr = open(join(self.params.download_path(), 'log.txt'), 'w+')
            Fido.fetch(self.fido_result, path=self.fits_folder,
                       downloader=Downloader(progress=True, file_progress=False, max_conn=20,
                                             overwrite=False))
            # sys.stderr = err
            print("  Success!")
        except Exception as e:
            print('1 + ', e)
    
    def validate_download(self):
        try:
            self.set_output_paths()
        except:
            set_output_paths(self)
        self.list_requested_files()
        self.local_fits_paths = list_files_in_directory(self.fits_folder)
        if self.params.delete_old():
            self.remove_all_old_fits_pngs()
            self.remove_all_old_pngs()
        
        working = False
        if working:
            self.validate_fits()
            self.redownload_bad_fits()
        
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
        png_directory = join(self.params.img_directory(), self.current_wave, 'png')
        got_png = list_files_in_directory(png_directory, 'png')
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
