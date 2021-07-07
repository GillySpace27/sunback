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
from utils.file_util import discover_best_data_directory
import astropy.units as u

from sunpy.instr.aia import aiaprep

import datetime


# aia1 = sunpy.map.Map(file_download[0])
# aia = aiaprep(aia1)

class FidoFetcher(Fetcher):
    """Gets some data"""
    wavelengths = ['0171', '0193', '0211', '0304', '0131', '0335', '0094']
    
    def __init__(self, params, base_url=None, base_dir_path=discover_best_data_directory()):
        self.beginTime = time()
        
        # Parse Inputs
        self.params = params
        self.params.archive_url(base_url)
        self.params.download_path(base_dir_path)
        
        if self.params.do_one():
            self.waves_to_do = [self.params.do_one()]
        else:
            self.waves_to_do = self.wavelengths
        
        # Set class variables
        self.local_wave_directory = None
        self.image_folder = None
        self.movie_folder = None
        self.fits_folder = None
        
        self.fido_result = None
        self.fido_num = None
        
        self.select_time_range()
        
        self.local_fits_paths = []
        self.requested_files = []
        self.redownload = []
        self.file_size_mode = None
        
        self.final_fits_paths = []
    
    def select_time_range(self):
        """Define Time Range, on the hour"""
        
        if self.params.do_recent():
            self.get_recent_range()
        else:
            self.get_time_range()
    
    def set_start_time(self, start_struct):
        try:
            self.start_time = start_struct.strftime('%Y/%m/%d %H:%M')
            self.start_time_long = int(start_struct.strftime('%Y%m%d%H%M%S'))
        except AttributeError:
            self.start_time = strftime('%Y/%m/%d %H:%M', start_struct)
            self.start_time_long = int(strftime('%Y%m%d%H%M%S', start_struct))
        
        self.start_string = self.parse_time_string_to_local(str(self.start_time_long), 2)[0]
    
    def set_end_time(self, end_struct):
        try:
            
            self.end_time = end_struct.strftime('%Y/%m/%d %H:%M')
            self.end_time_long = int(end_struct.strftime('%Y%m%d%H%M%S'))
        except AttributeError:
            self.end_time = strftime('%Y/%m/%d %H:%M', end_struct)
            self.end_time_long = int(strftime('%Y%m%d%H%M%S', end_struct))
        
        self.end_time_string = self.parse_time_string_to_local(str(self.end_time_long), 2)[0]
    
    def get_time_range(self):
        start, end = self.params.time_period()
        
        start_struct = datetime.datetime.strptime(start, '%Y/%m/%d %H:%M')
        end_struct = datetime.datetime.strptime(end, '%Y/%m/%d %H:%M')
        
        self.set_start_time(start_struct)
        self.set_end_time(end_struct)
    
    def get_recent_range(self):
        # Get the Start Time
        current_time = time() + timezone
        start_list = list(localtime(current_time - (self.params.range() + 2 / 24) * 60 * 60 * 24))
        start_list[4] = 0  # Minutes
        start_list[5] = 0  # Seconds
        start_struct = struct_time(start_list)
        self.set_start_time(start_struct)
        
        # Get the Current Time
        now_list = list(localtime(current_time - 2 * 60 * 60))
        now_list[4] = 0  # Minutes
        now_list[5] = 0  # Seconds
        end_struct = struct_time(now_list)
        self.set_end_time(end_struct)
    
    @staticmethod
    def parse_time_string_to_local(downloaded_files, which=0, local=True):
        if which == 0:
            time_string = downloaded_files[0][-25:-10]
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = int(time_string[9:11])
            minute = time_string[11:13]
        elif which == 3:
            time_string = downloaded_files
            split = time_string.split("_")
            # import pdb; pdb.set_trace()
            year = split[3]
            month = split[4]
            day = split[5].split('t')[0]
            hour_raw = split[5].split('t')[1]
            minute = split[6]
        else:
            time_string = downloaded_files
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = time_string[8:10]
            minute = time_string[10:12]
        
        struct_time = (int(year), int(month), int(day), int(hour_raw), int(minute), 0, 0, 0, -1)
        # print(struct_time)
        
        if local:
            theTime = localtime(timegm(struct_time))
        else:
            theTime = struct_time
        
        new_time_string = strftime("%I:%M%p %m/%d/%Y", theTime).lower()
        if new_time_string[0] == '0':
            new_time_string = new_time_string[1:]
        
        # print(year, month, day, hour, minute)
        # new_time_string = "{}:{}{} {}/{}/{} ".format(hour, minute, suffix, month, day, year)
        time_code = strftime("%Y%m%d%I%M%S", theTime)
        
        return new_time_string, time_code
    
    def fetch(self):
        """ Find the Most Recent Images """
        if not self.params.download_images():
            print("Using Local Files... \n")
        
        for self.current_wave in self.waves_to_do:
            self.build_paths()
            self.fido_get_fits()
            # self.set_fits_list()
        self.set_output_paths()
    
    # def set_fits_list(self):
    #     abs_paths = [join(self.fits_folder, st) for st in self.local_fits_paths]
    #     self.final_fits_paths.extend(abs_paths)
    #     print("file box= {}".format(len(self.final_fits_paths)))
    # local_paths = self.params.local_fits_paths()
    # local_paths.extend()
    
    def fido_get_fits(self):
        if self.params.download_images():
            self.download_fits_series()
            self.validate_download()
  
    def set_output_paths(self):
        self.params.local_fits_paths(self.final_fits_paths)
        # for wave in self.waves_to_do:
        fit_folder = join(self.params.download_path(), self.current_wave, 'fits')
        list_of_files = self.list_files_in_directory(fit_folder)
        abs_paths = [join(fit_folder, ff) for ff in list_of_files]
        self.final_fits_paths.extend(abs_paths)
    
    def build_paths(self):
        """Make the file structure to hold the images"""
        self.local_wave_directory = join(self.params.download_path(), self.current_wave)
        self.fits_folder = join(self.local_wave_directory, "fits\\")
        self.image_folder = join(self.local_wave_directory, "png\\")
        self.movie_folder = abspath(join(self.local_wave_directory, "..\\movies\\"))
        makedirs(self.local_wave_directory, exist_ok=True)
        makedirs(self.image_folder, exist_ok=True)
        makedirs(self.movie_folder, exist_ok=True)
    
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
                                       attrs.Sample(self.params.cadence_minutes())) #, attrs.Resolution(self.params.resolution()))  # , a.vso.Provider('jsoc'))
        self.fido_num = self.fido_result.file_num
        print("Found {}".format(self.fido_num))
    
    def fido_parse_result(self):
        """Examine the search results"""
        
        first_result = self.fido_result.get_response(0)[0]
        last_result = self.fido_result.get_response(0)[-2]
        
        self.name = first_result.wave.wavemin
        
        time_start = first_result.time.start
        time_end = last_result.time.end
        
        begin_time = self.parse_time_string_to_local(''.join(i for i in time_start if i.isdigit()), 2)[0]
        end_time = self.parse_time_string_to_local(''.join(i for i in time_end if i.isdigit()), 2)[0]
        
        self.extra_string = "from {} to {}".format(begin_time, end_time)
        
        self.startTime = time_start  # #self.parse_time_string_to_local(str(time_start.fits), 2)[0]
        self.endTime = time_end  # ''.join(i for i in time_end.iso if i.isdigit())   #self.parse_time_string_to_local(str(time_end.fits), 2)[0]
        
        print("    Search Found {} Images {}...".format(self.fido_num, self.extra_string), flush=True)
        
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
            print("Success!")
        except Exception as e:
            print('1 + ', e)
    
    def validate_download(self):
        self.set_output_paths()
        self.list_requested_files()
        self.local_fits_paths = self.list_files_in_directory()
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
    
    def list_files_in_directory(self, directory=None, extension="fits"):
        directory = self.fits_folder if directory is None else directory
        makedirs(directory, exist_ok=True)
        return [f.casefold() for f in listdir(directory) if f.endswith('.' + extension)]
        
        # files = listdir(self.fits_folder)
        # self.already_downloaded = []
        # for filename in files:
        #     if filename.endswith(".fits"):
        #         self.already_downloaded.append(filename.casefold()) #localtime
        # try:
        #     timeString = int(self.time_from_filename(filename, local=False)[1][:-2])
        # except Exception as e:
        #     timeString = filename[3:11] + filename[12:16]
        # print(str(timeString)[-4:])
    
    def remove_all_old_pngs(self):
        requested_pngs = [x.replace('fits', 'png') for x in self.local_fits_paths]
        png_directory = join(self.params.download_path(), self.current_wave, 'png')
        got_png = self.list_files_in_directory(png_directory, 'png')
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
            print("Success! Deleted {} old images".format(len(self.redownload)))
    
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
                        hh=0
                        total_counts = np.nansum(hdul[hh].data)
                    except Exception as e:
                        print(e)
                        hh=1
                        delete = True
                        total_counts = np.nansum(hdul[hh].data)
                        delete = False
                    this_size = stat(abs_path).st_size
                    data = hdul[hh].data
                    if total_counts < 0: # or not this_size == self.file_size_mode:
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
