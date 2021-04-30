import urllib
from datetime import datetime
from urllib.request import urlretrieve
import numpy as np
import requests
from bs4 import BeautifulSoup
from utils.file_util import discover_best_data_directory
from fetch.Fetch import Fetch
from tqdm import tqdm

mr_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images

class WebFetch(Fetch):
    
    def __init__(self, base_url=mr_url, base_dir_path=discover_best_data_directory()):
        self.data_path = base_dir_path
        self.url = base_url
    
    def download_fits_files(self, url=mr_url):
        """Processes the img series"""
        print("\nDownloading Images...", flush=True)
        paths = []
        tries = 3
        
        for link in tqdm(self.__get_fits_links(url)):
            # For each image
            local_path = self.data_path + link.split('/')[-1]  # TODO make a better local path
            
            for ii in np.arange(tries):
                # Retry download
                try:
                    urlretrieve(link, local_path)
                    break
                except urllib.error.ContentTooShortError:
                    print("Failed Download...Retrying {} / {}".format(ii, tries))
                    pass
            
            paths.append(local_path)
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
        img_links = [mr_url + link['href'] for link in links if link['href'].endswith('fits')]
        img_links = [lnk for lnk in img_links if '4500' not in lnk]
        return img_links
