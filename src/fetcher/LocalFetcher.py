import urllib
from datetime import datetime
from urllib.request import urlretrieve
import numpy as np
import requests
from bs4 import BeautifulSoup
from utils.file_util import discover_best_data_directory
from fetcher.Fetcher import Fetcher
from tqdm import tqdm
from os import listdir
from time import sleep
from os.path import join

archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


class LocalFetcher(Fetcher):
    
    def __init__(self, params, base_url=archive_url, base_dir_path=discover_best_data_directory()):
        self.params = params
        self.params.download_path(base_dir_path)
        self.params.time_path(base_dir_path + "\\image_times")
      
        
    def fetch(self):
        """Loads the img series from disk"""
        download_path = self.params.download_path()
        print("Loading Fits Files from {}...".format(download_path), end='', flush=True)

        all_paths = listdir(download_path)
        fits_paths = [join(download_path, path)
                      for path in all_paths if '.fits' in path]
        self.params.local_fits_paths(fits_paths)

        print("Success! {} Found\n".format(len(fits_paths)))
        sleep(1)
    
        
        
        
        
        
        