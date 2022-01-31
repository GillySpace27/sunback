import copy
import os
import shutil

from os import remove
from os.path import join, basename
from time import strptime, mktime
import sys

from drms import DrmsExportError
from parfive import Downloader
from sunpy.net import Fido, attrs
import numpy as np
from astropy.io import fits
from os import stat
from fetcher.Fetcher import Fetcher
from utils.time_util import parse_time_string_to_local
import astropy.units as u

import datetime
from utils.time_util import define_time_range, define_recent_range

default_base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images

global global_verb
global_verb = False


def vprint(in_string, verb=None, *args, **kwargs):
    global global_verb
    if verb is not None:
        global_verb = verb
    if FidoFetcher.verb or global_verb:
        print(in_string, *args, **kwargs)


class FidoFetcher(Fetcher):
    """Gets some data"""
    description = "Get Fits Files from the Internet using Fido"
    verb = True
    filt_name = "Fido Fetcher"
    num_files_needed = None
    batch_id = 0
    needed_files = None
    results = None
    temp_folder = None
    int_tm_tot = 0
    fits_path=None
    
    def __init__(self, params=None, quick=False, rp=None):
        # Initialize class variables
        super().__init__(params, quick, rp)

        self.reprocess_mode(rp)
        self.params.load_preset_time_settings()
        # self.fetch()
    
    ## Main Fetch Logic
    def fetch(self, params=None, quick=False, rp=None, verb=True):
        if verb is not None:
            self.verb = verb
        """ Find the Most Recent Images """
        self.__init__(params, quick, rp)
        # self.verb = True
        self.fido_get_fits(self.params.current_wave(), temp=self.params.do_temp)
    
    def cleanup(self):
        # self.fido_download_fits_ensured(hold=False, temp=True)
        # self.delete_temp_folder_items(delete_folder_too=True)
        pass
    
    def enumerate(self):
        # for fits_path in self.params.local_fits_paths():
        #     print(fits_path)
        pass
    
    def fido_get_fits(self, current_wave, temp=False):
        self.load(self.params, wave=current_wave)
        # vprint("\r          ")
        vprint("v Fetching Fits Files: {}".format(self.params.current_wave()), self.verb)
        if self.params.download_files() or self.reprocess_mode() or not self.verb:
            self.print_load_banner(verb=self.verb)
            self.download_fits_series(temp=temp)
            # self.validate_download()
            # self.enumerate()
        else:
            vprint(" ^ Using {} Cached Fits Files\n".format(self.params.n_fits), self.verb)
    
    def download_fits_series(self, temp=True, hold=None):
        if hold is None:
            hold = False  # TODO Fix this
        self.define_range()
        self.fido_check_for_fits()
        if self.fido_search_found_num:
            self.fido_parse_result()
            self.fido_download_fits_ensured(temp, hold)
        else:
            print("\n     No Images Found\n")
    
    def fido_check_for_fits(self, verb=None):
        """Find the science images"""
        self.verb = self.verb or verb
        vprint("\n *   Looking for Images of {} from {} to {}...".format(
                self.params.current_wave(), self.start_time_string, self.end_time_string), flush=True, end='', verb=self.verb)
        jsoc_email = "chris.gilly@colorado.edu"
        
        # Make the base required attributes
        time_attr = attrs.Time(self.start_time, self.end_time)
        wave_attr = attrs.Wavelength(int(self.params.current_wave()) * u.angstrom)
        sample_attr = attrs.Sample(self.params.cadence_minutes())
        base_attrs = time_attr & wave_attr & sample_attr
        
        # from sunpy.net.attrs import Instrument
        # from sunpy.net.jsoc.attrs import Keys
        # Search for records from the internet
        
        attrs.jsoc.Keys
        
        if self.params.do_recent():
            inst_attr = attrs.Instrument.aia
        else:
            inst_attr = attrs.jsoc.Series.aia_lev1_euv_12s & \
                        attrs.jsoc.Notify(jsoc_email) & \
                        attrs.jsoc.Segment.image
                        
            
        fido_search_result = Fido.search(base_attrs, inst_attr)
        self.fido_search_result = fido_search_result
        self.fido_search_found_num = self.fido_search_result.file_num
    
    def fido_parse_result(self):
        """Examine the search results"""
        self.start_time, self.end_time = self.get_start_and_end_times_from_result()
        
        begin_time = parse_time_string_to_local(self.start_time, 4)[0]
        end_time = parse_time_string_to_local(self.end_time, 4)[0]
        self.extra_string = "from {} to {}".format(begin_time, end_time)
        
        if self.fido_search_found_num > 1:
            vprint("\n *      Search Found {: 3} Images {}...".format(
                    self.fido_search_found_num, self.extra_string), flush=True, verb=self.verb)
        elif self.fido_search_found_num == 1:
            vprint("\n *      Search Found  {: 3} Image  at  {}...".format(
                    1, begin_time), flush=True, verb=self.verb)
            vprint(" *                             End = {}...".format(
                    self.end_time_string), flush=True, verb=self.verb)
        else:
            vprint("\n *      Search Found Nothing")
            raise FileNotFoundError
        
        while len(self.name) < 4:
            self.name = '0' + self.name
            
        if self.fido_search_found_num > 200 and False:
            response = input("Do you still want to download all {} images? [y]/n > ".format(self.fido_search_found_num))
            if 'n' in response.casefold():
                print("Stopping!")
                raise StopIteration
            print("Continuing. ", end='')
            
    def store_requests(self):
        try:
            response = self.fido_search_result.get_response(0)
        except AttributeError:
            response = self.fido_search_result
        self.needed_files = response
        self.num_files_needed = len(self.needed_files)
    
    def fido_download_fits_ensured(self, temp=False, hold=False):
        """Download the files from fido_search_result"""
        
        SubDownloader = Downloader(progress=True, file_progress=False, max_conn=20,
                                   overwrite=False)
        
        self.out_path = self.params.temp_directory() if temp else self.params.fits_directory()
        
        self.store_requests()
        
        main_stdout = sys.stdout
        
        if not hold:
            loc = os.path.join(self.params.temp_directory(), 'log.txt')
            # with open(loc, mode="w+") as sys.stdout:
            self.verb = False
            print("Fido Fetching...")
            try:
                results = Fido.fetch(self.needed_files, path=self.out_path, downloader=SubDownloader)
            except DrmsExportError as e:
                print(e)
                results = []
                
            self.n_fits = len(results)+0
            # if ensured:
            #     results = self.fido_multi_download()
            self.multi_banner()
            self.results = copy.copy(results)
            sys.stdout = main_stdout
            return self.results
    
    def fido_multi_download(self):
        self.n_fits = len(self.results)
        
        while self.n_fits != self.fido_search_found_num:
            self.results = Fido.fetch(self.results, path=self.out_path)
            self.n_fits = len(self.results)
        self.n_fits = len(self.results)
        
        return self.results
    
    # Time Related Things #########################################
    
    def get_start_and_end_times_from_result(self):
        # self.verb = True
        all_times = self.fido_search_result.get_response(0)
        start_time_list = []
        # end_time_list = []
        for result in all_times:
            try:
                start_time_list.append(result["T_REC"])
            except KeyError:
                start_time_list.append(result["Start Time"].value)
            
            # end_time_list.append(result.time.end)
        
        times = sorted(start_time_list)
        time_start = times[0]
        time_end = times[-1]
        # ii=0
        # while time_start[-3:-1] < self.start_time[-2:]:
        #     time_start = times[ii]
        #     ii+=1
        # for t in range(ii-1):
        #     self.fido_search_result[0].remove_row(0)
        self.fido_search_found_num = self.fido_search_result.file_num
        
        return time_start, time_end
    
    def define_range(self):
        """Defines the time range of imagery desired"""
        if self.params.do_recent():
            self.unpack_time_strings(*define_recent_range(self.params.range()))
        else:
            self.unpack_time_strings(*define_time_range(*self.params.time_period()))
    
    def unpack_time_strings(self, start, end):
        """Unpacks the time lists"""
        self.start_time, self.start_time_long, self.start_time_string = start
        self.end_time, self.end_time_long, self.end_time_string = end
    
    # Printing #####################################################
    
    def multi_banner(self):
        if self.n_fits == self.fido_search_found_num:
            print(" ^     Successfully Downloaded all {} Files\n".format(self.n_fits), flush=True)
        elif self.n_fits:
            print(" ^     Downloaded {} Files out of {}\n".format(self.n_fits, self.fido_search_found_num), flush=True)
        else:
            print(" ^     Unable to Download...Try again Later.")
            raise (FileNotFoundError(" Unable to Download...Try again Later."))
    
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
