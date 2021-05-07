import urllib
from datetime import datetime
from urllib.request import urlretrieve
import numpy as np
import requests
from bs4 import BeautifulSoup

from executor.Executor import Executor
from utils.file_util import discover_best_data_directory
from fetcher.Fetcher import Fetcher
from tqdm import tqdm
from os import listdir
from time import sleep
from os.path import join

archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


class LocalExecutor(Executor):
    
    def __init__(self, params, base_url=archive_url, base_dir_path=discover_best_data_directory()):
        self.params = params
        self.params.download_path(base_dir_path)
        
    def execute(self):
        """Loads the img series from disk"""
        download_path = self.params.download_path()
        print("Loading PNGs from {}...".format(download_path), end='', flush=True)

        all_paths = listdir(download_path)
        png_paths = [join(download_path, path)
                      for path in all_paths if '.png' in path[-4:]]
        self.params.local_img_paths(png_paths)

        print("Success! {} Found\n".format(len(png_paths)))
        sleep(1)
    
        
        
        
        
        
