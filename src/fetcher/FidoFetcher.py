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
    filt_name = "Fido Fetcher"
    
    def __init__(self, params=None, quick=False, rp=None):
        # Initialize class variables
        super().__init__(params, quick, rp)
        self.num_files_needed = None
        self.batch_id = 0
        self.list_of_needed_files = None
        self.results = None
        self.temp_folder = None
        self.reprocess_mode(rp)
        self.int_tm_tot = 0
        self.params.load_preset_time_settings()
        # self.fetch()
    
    ## Main Fetch Logic
    def fetch(self, params=None, quick=False, rp=None, verb=True):
        if verb is not None:
            self.verb = verb
        """ Find the Most Recent Images """
        self.__init__(params, quick, rp)
        # self.verb = True
        self.fido_get_fits(self.params.current_wave())
    
    def cleanup(self):
        self.fido_download_fits_ensured(hold=False, temp=True)
        pass
        # self.delete_temp_folder_items(delete_folder_too=True)
    
    def fido_get_fits(self, current_wave):
        self.load(self.params, wave=current_wave)
        vprint(" v Fetching Fits Files: {}".format(self.params.current_wave()), self.verb)
        if self.params.download_files() or self.reprocess_mode() or not self.verb:
            self.print_load_banner(verb=self.verb)
            self.download_fits_series()
            self.validate_download()
        else:
            vprint(" ^ Using {} Cached Fits Files\n".format(self.params.n_fits), self.verb)
    
    def download_fits_series(self):
        self.define_range()
        self.fido_check_for_fits()
        if self.fido_search_found_num:
            self.fido_parse_result()
            self.fido_download_fits_ensured()
        else:
            print("\n     No Images Found\n")
    
    def fido_check_for_fits(self):
        """Find the science images"""
        vprint("\n *   Looking for Images of {} from {} to {}...".format(
                self.params.current_wave(), self.start_time_string, self.end_time_string), flush=True, end='', verb=self.verb)
        jsoc_email = "chris.gilly@colorado.edu"
        # Search for records from the internet
        self.fido_search_result = Fido.search(attrs.Time(self.start_time, self.end_time),
                                              attrs.Wavelength(int(self.params.current_wave()) * u.angstrom),
                                              attrs.Sample(self.params.cadence_minutes()),
                                              attrs.jsoc.Series.aia_lev1_euv_12s,
                                              attrs.jsoc.Notify(jsoc_email),
                                              attrs.jsoc.Segment.image,
                                              )
        
        # , attrs.Resolution(self.params.resolution()))  # , a.vso.Provider('jsoc'))
        
        self.fido_search_found_num = self.fido_search_result.file_num
        
        # print("Found {}".format(self.fido_search_found_num))
        
        # self.exposure_paths = self.get_exposure_paths()
        # vprint(" *     Successfully Downloaded {} Files\n".format(len(self.exposure_paths)), flush=True, verb=self.verb)
        # vprint(" * File Location: {}".format(self.params.temp_directory()))
        # sys.stdout.flush()
    
    def get_start_and_end_times(self):
        # self.verb = True
        all_times = self.fido_search_result.get_response(0)
        start_time_list = []
        # end_time_list = []
        for result in all_times:
            start_time_list.append(result["T_REC"])
            # end_time_list.append(result.time.end)
        
        times = start_time_list
        time_start = (min(times))
        time_end = (max(times))
        return time_start, time_end
        
        # first_result = self.fido_search_result.get_response(0)[0]
        # last_result = self.fido_search_result.get_response(0)[-1]
        #
        # self.name = first_result.wave.wavemin
        #
        # first_start_int = int(first_result.time.start)
        # last_start_int  = int( last_result.time.start)
        # first_end_int   = int(first_result.time.start)
        # last_end_int    = int( last_result.time.start)
        #
        # times = [first_start_int, last_start_int, first_end_int, last_end_int]
    
    def fido_parse_result(self):
        """Examine the search results"""
        self.start_time, self.end_time = self.get_start_and_end_times()
        
        begin_time = parse_time_string_to_local(self.start_time, 4)[0]
        end_time = parse_time_string_to_local(self.end_time, 4)[0]
        self.extra_string = "from {} to {}".format(begin_time, end_time)
        
        vprint("\n *      Search Found {: 3} Images {}...".format(
                self.fido_search_found_num, self.extra_string), flush=True, verb=self.verb)
        
        while len(self.name) < 4:
            self.name = '0' + self.name
    
    def store_requests(self):
        if self.list_of_needed_files is None:
            self.list_of_needed_files = self.fido_search_result.get_response(0)
            self.fetch_box = [self.fido_search_result.get_response(0)]
        else:
            for row in self.fido_search_result.get_response(0):
                row.id = self.batch_id
                self.list_of_needed_files.add_row(row)
            self.fetch_box.append(self.fido_search_result.get_response(0))
        self.batch_id += 1
        
        # n_new = self.fido_search_found_num
        self.num_files_needed = self.list_of_needed_files.file_num = len(self.list_of_needed_files)

    def fido_download_fits_ensured(self, ensured=False, temp=False, hold=False):
        """Download the files from fido_search_result"""
        
        SubDownloader = Downloader(progress=True, file_progress=False, max_conn=20,
                                   overwrite=False)
        
        self.out_path = self.temp_folder if temp else self.fits_folder
        
        self.store_requests()
        if not hold:
            self.results = Fido.fetch(self.list_of_needed_files, path=self.out_path, downloader=SubDownloader)
            self.n_fits = len(self.results)
            
            if ensured:
                self.results = self.fido_multi_download()
            
            self.multi_banner()
            return self.results
    
    def fido_multi_download(self):
        self.n_fits = len(self.results)

        while self.n_fits != self.fido_search_found_num:
            self.results = Fido.fetch(self.results, path=self.out_path)
            self.n_fits = len(self.results)
        self.n_fits = len(self.results)

        return self.results
    
    def multi_banner(self):
        if self.n_fits == self.fido_search_found_num:
            print(" ^     Successfully Downloaded all {} Files\n".format(self.n_fits), flush=True)
        elif self.n_fits:
            print(" ^     Downloaded {} Files out of {}\n".format(self.n_fits, self.fido_search_found_num), flush=True)
        else:
            print(" ^     Unable to Download...Try again Later.")
            raise (FileNotFoundError(" Unable to Download...Try again Later."))

    def define_range(self):
        """Defines the time range of imagery desired"""
        if self.params.do_recent():
            self.parse_time(*define_recent_range(self.params.range()))
        # elif self.params.exposure_time_seconds():
        #     self.parse_time(*define_duration_range(*self.params.time_period()))
        else:
            start, end = self.params.time_period()
            self.parse_time(*define_time_range(start, end))
    
    def parse_time(self, start, end):
        """Unpacks the time lists"""
        self.start_time, self.start_time_long, self.start_time_string = start
        self.end_time, self.end_time_long, self.end_time_string = end

        
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
        # self.fido_download_fits_ensured()
        
        # self.find_missing_images()
        # self.get_missing_images()
    
    def list_requested_files(self):
        self.requested_files = []
        self.requested_response = []
        for ii in np.arange(self.fido_search_found_num):
            self.requested_files.append(self.fido_search_result.get_response(0)[ii]['fileid'].casefold())
            self.requested_files.append(self.fido_search_result.get_response(0)[ii]['time']['start_timestamp'])
    
    @staticmethod
    def parse_filename_to_time(local_file):
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
    
    @staticmethod
    def out_of_range(hdul):
        print('A')
        pass
        return False
    









































































