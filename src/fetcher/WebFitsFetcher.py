import sys
import urllib
from datetime import datetime
from os import rename, remove
from os.path import exists, join
from urllib.request import urlretrieve
import numpy as np
import requests
from bs4 import BeautifulSoup
# from utils.file_util import discover_best_data_directory
from fetcher.Fetcher import Fetcher
from tqdm import tqdm



class WebFitsFetcher(Fetcher):
    base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images
    description = "Get Fits Files from {}".format(base_url)
    def fetch(self, params=None):
        """Gets the Fits Files from the Archive URL
        :param params:
        """
        if params is not None:
            self.params = params
        # self.params.current_wave()
        self.load(params, quietly=True, wave=self.params.current_wave('rainbow'))
        if self.params.redownload_files():
            print("  Downloading Fits Files from {}...".format(self.base_url), flush=True)
            # super.super.__init__(params)
            img_links = self.__get_fits_links(self.base_url)
            paths = []
            for link in tqdm(img_links, desc="  "):
                paths.append(self.grab(link))
        
            self.__get_img_time()
            sys.stdout.flush()
            print(" *  Successfully Downloaded {} Files\n".format(len(paths)), flush=True)
            return paths
        return self.params.local_fits_paths()
    
    def grab(self, link):
        tries = 3
        filename = link.split('/')[-1]
        local_path = join(self.params.fits_directory(), filename)
        local_temp_path = join(self.params.fits_directory(), "download__" + filename)
        for ii in np.arange(tries):
            # Retry download
            try:
                urlretrieve(link, local_temp_path)
                if exists(local_path):
                    remove(local_path)
                rename(local_temp_path, local_path)
                break
            except urllib.error.ContentTooShortError:
                print("Failed Download...Retrying {} / {}".format(ii, tries))
                pass
        
        # paths.append(local_path)
    
    @staticmethod
    def __get_fits_links(url):
        """gets the list of files to pull"""
        # create response object
        r = requests.get(url)
        
        # create beautiful-soup object
        soup = BeautifulSoup(r.content, 'html5lib')
        
        # not_wanted all links on web-page
        links = soup.findAll('a')
        
        # filter the link sending with .fits
        img_links = [url + link['href'] for link in links if link['href'].endswith('fits')]
        img_links = [lnk for lnk in img_links if '4500' not in lnk]
        return img_links
    
    def __get_img_time(self):
        """Gets the time file"""
        image_time = requests.get(self.base_url + "image_times").text[9:25]
        with open(self.params.time_path(), 'w') as fp:
            fp.write(image_time)
