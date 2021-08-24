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

base_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images


class LocalFetcher(Fetcher):
    
    def __init__(self, params, base_url=base_url, base_directory=None):
        self.params = params
        self.params.build_paths_single(base_url, base_directory)
    
    def fetch(self):
        self.load_fits()
        self.load_imgs()


class LocalFitsFetcher(LocalFetcher):
    def fetch(self):
        self.load_fits()


class LocalImgFetcher(LocalFetcher):
    def fetch(self):
        self.load_imgs()
