import sys
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

default_base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


class LocalFetcher(Fetcher):
    description = "Load the images from Disk"
    
    def fetch(self, params=None):
        print("  Loading Local Files...", end='')
        self.load(params)
        num = self.params.n_fits
        print("     Successfully Loaded {}".format(num) if num>0 else "No Files to Load!")
        print()
        if num == 0:
            # self.params.fetchers([self.params.alternate])
            print("\n    !!Quitting Program!!")
            sys.exit(1)
