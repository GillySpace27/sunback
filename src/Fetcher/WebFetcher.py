import urllib
from datetime import datetime
from os import rename, remove
from urllib.request import urlretrieve
import numpy as np
import requests
from bs4 import BeautifulSoup
from utils.file_util import discover_best_data_directory
from Fetcher.Fetcher import Fetch
from tqdm import tqdm

archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


class WebFetcher(Fetch):
    
    def __init__(self, params, base_url=archive_url, base_dir_path=discover_best_data_directory()):
        self.params = params
        self.params.archive_url(base_url)
        self.params.download_path(base_dir_path)
        self.params.time_path(base_dir_path + "\\image_times")
    
    def fetch(self, url=archive_url):
        """Processes the img series"""
        print("Downloading Fits Files from {}...".format(self.params.archive_url()), flush=True)
        paths = []
        tries = 3
        
        for link in tqdm(self.__get_fits_links(url)):
            # For each image
            local_path = self.params.download_path() + '\\' + link.split('/')[-1]  # TODO make a better local path
            local_temp_path = self.params.download_path() + '\\' + "download__" + link.split('/')[-1]
            for ii in np.arange(tries):
                # Retry download
                try:
                    urlretrieve(link, local_temp_path)
                    remove(local_path)
                    rename(local_temp_path, local_path)
                    break
                except urllib.error.ContentTooShortError:
                    print("Failed Download...Retrying {} / {}".format(ii, tries))
                    pass
            
            paths.append(local_path)
        
        self.__get_img_time()
        self.params.local_fits_paths(paths)
        print("Success!\n", flush=True)
        return paths
    
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
        img_links = [archive_url + link['href'] for link in links if link['href'].endswith('fits')]
        img_links = [lnk for lnk in img_links if '4500' not in lnk]
        return img_links
    
    def __get_img_time(self):
        """Gets the time file"""
        image_time = requests.get(archive_url + "image_times").text[9:25]
        with open(self.params.time_path(), 'w') as fp:
            fp.write(image_time)
