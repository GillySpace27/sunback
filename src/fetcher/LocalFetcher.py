import sys
# import urllib
# from datetime import datetime
# from urllib.request import urlretrieve
# import numpy as np
# import requests
# from bs4 import BeautifulSoup
# from utils.file_util import find_root_directory
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
        print(" ^    Successfully Discovered {} fits and {} images\n".format(self.n_fits, self.n_imgs)
              if num>0 else "No Files to Load!")
        if num == 0:
            # self.params.fetchers([self.params.alternate])
            print("\n    !!Quitting Program!!\n")
            print(self.params.base_directory())
            print(self.params.fits_directory())
            print(self.params.imgs_directory())
            
            sys.exit(1)

class LocalSingleFetcher(Fetcher):
    description = "Load the image from Disk"
    filt_name = "Local Single Fetcher"
    
    def fetch(self, params=None):
        print(" v Loading Local File...")
        self.load(params)
        for self.params.hdu_name in self.params.start_frame:
            try:
                self.load_fits_image(self.params.use_image_path(), self.params.hdu_name)
                print("Using {} HDU".format(self.params.hdu_name))
                break
            except Exception as e:
                print("LocalSingleFetcher")
                print(e)
                
        import os.path as path
        print("     ", path.dirname(self.params.use_image_path()))
        print("     ", path.basename(self.params.use_image_path()))
        # self.view_original()
        print(" ^ Success!")
    

        # plt.show()
        
        
        # self.params.img_stuff = self.load_first_fits_field()
        # self.params.original_image, wave, t_rec, center, int_time = self.params.img_stuff
        # self.params.modified_image = self.params.original_image + 0
        # self.params.set_current_wave(wave)

        
        # self.load(params)
        # num = self.n_fits + self.n_imgs
        # print(" ^    Successfully Discovered {} fits and {} images\n".format(self.n_fits, self.n_imgs)
        #       if num>0 else "No Files to Load!")
        # if num == 0:
        #     # self.params.fetchers([self.params.alternate])
        #     print("\n    !!Quitting Program!!\n")
        #     print(self.params.base_directory())
        #     print(self.params.fits_directory())
        #     print(self.params.imgs_directory())
        #
        #     sys.exit(1)
