import urllib
from datetime import datetime
from os import rename, remove
from os.path import exists, join
from urllib.request import urlretrieve
import numpy as np
import requests
from bs4 import BeautifulSoup
from utils.file_util import discover_best_data_directory
from fetcher.Fetcher import Fetcher
from tqdm import tqdm

base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


class WebFitsFetcher(Fetcher):
    
    def __init__(self, params, base_url=base_url, base_directory=None):
        self.params = params
        self.params.build_paths_single(base_url, base_directory)
    
    def fetch(self):
        """Gets the Fits Files from the Archive URL"""
        print("  Downloading Fits Files from {}...".format(self.params.archive_url()), flush=True)
        img_links = self.__get_fits_links(self.params.archive_url())
        paths = []
        for link in tqdm(img_links):
            paths.append(self.grab(link))
            
        self.__get_img_time()
        self.load_fits()
        print("Success!\n", flush=True)
        return paths
    
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
        
        # find all links on web-page
        links = soup.findAll('a')
        
        # filter the link sending with .fits
        img_links = [url + link['href'] for link in links if link['href'].endswith('fits')]
        img_links = [lnk for lnk in img_links if '4500' not in lnk]
        return img_links
    
    def __get_img_time(self):
        """Gets the time file"""
        image_time = requests.get(base_url + "image_times").text[9:25]
        with open(self.params.time_path(), 'w') as fp:
            fp.write(image_time)
