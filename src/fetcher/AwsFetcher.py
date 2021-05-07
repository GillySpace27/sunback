import urllib
from datetime import datetime
from os import rename, remove
from os.path import join
from urllib.request import urlretrieve
import numpy as np
import requests
from bs4 import BeautifulSoup
from utils.file_util import discover_best_data_directory
from fetcher.Fetcher import Fetcher
from tqdm import tqdm
import boto3
import os

class AwsFetcher(Fetcher):
    
    def __init__(self, params, base_url=None, base_dir_path=discover_best_data_directory()):
        self.params = params
        self.params.archive_url(base_url)
        self.params.download_path(base_dir_path)
        self.params.time_path(base_dir_path + "\\image_times")
    
    def fetch(self):
        """Get the PNGs from the S3 Bucket"""
        s3_resource = boto3.resource('s3')
        my_bucket = s3_resource.Bucket('gillyspace27-test-billboard')
        objects = my_bucket.objects.filter(Prefix='renders/')
        local_dir = self.params.download_path()
        print("\nDownloading PNGs from S3 to {}".format(local_dir))
        fileBox = []
        for obj in objects:
            path, filename = os.path.split(obj.key)
            if 'orig' in obj.key or 'archive' in obj.key or "thumbs" in obj.key or "4500" in obj.key:
                continue
            if self.params.do_one() and self.params.do_one() not in obj.key:
                continue
            print('    ', filename)
            loc = join(local_dir, filename)
            my_bucket.download_file(obj.key, loc)
            fileBox.append(loc)
        print("All Downloads Complete\n")
        return fileBox
    
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
