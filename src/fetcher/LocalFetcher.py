import sys
# import urllib
# from datetime import datetime
# from urllib.request import urlretrieve
# import numpy as np
# import requests
# from bs4 import BeautifulSoup
# from utils.file_util import discover_best_data_directory
from fetcher.Fetcher import Fetcher
# from tqdm import tqdm
# from os import listdir
# from time import sleep
# from os.path import join

default_base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


class LocalFetcher(Fetcher):
    description = "Load the images from Disk"
    filt_name = "Local Fetcher"
    
    def fetch(self, params=None):
        print(" v Loading Local Files...")
        self.load(params)
        num = self.n_fits + self.n_imgs
        print(" ^    Successfully Discovered {} fits and {} images\n".format(self.n_fits, self.n_imgs) if num>0 else "No Files to Load!")
        if num == 0:
            # self.params.fetchers([self.params.alternate])
            print("\n    !!Quitting Program!!\n")
            print(self.params.base_directory())
            print(self.params.fits_directory())
            print(self.params.imgs_directory())
            
            sys.exit(1)
