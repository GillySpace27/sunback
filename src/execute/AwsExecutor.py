from datetime import datetime
from os import makedirs
from os.path import abspath
from platform import system
from time import sleep, time

import boto3
import requests
from PIL import Image
from astropy.io import fits
from bs4 import BeautifulSoup
from tqdm import tqdm

from execute.Executor import Executor
from science.modify import Modify
from utils.file_util import discover_best_data_directory

# Flags
set_local_background = False

# Select Amazon Resources
s3 = boto3.resource('s3')
bucket = s3.Bucket('gillyspace27-test-billboard')
s3_client = boto3.client('s3')

# Location of the Solar Images
archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"

# Initialization
last_time = time()
start_time = last_time

default_sleep = 30


# Web Version
class AwsExecutor(Executor):
    def __init__(self, params):
        self.params = params
        self.data_path = discover_best_data_directory()
        print("AWS EXECUTOR")
    
    def execute(self, paths):
        """Loop over the wavelengths and normalize, then wait"""
        
        self.modify_upload_img_series(paths)
        self.sleep_until_delay_elapsed()

    
    def modify_upload_img_series(self, paths):
        """Processes the img series"""
        print("\nProcessing Images...", flush=True)
        self.save_times()
        for path in tqdm(paths):
            with fits.open(path) as hdul:
                img_paths = self.modify_img(hdul)
                self.upload(img_paths)
                
    
    def save_times(self):
        """Saves the Time file to S3 so we know when images were taken"""
        image_times = requests.get(archive_url + "image_times").text[9:25]
        
        path = self.data_path + "\\image_times"
        with open(path, 'w') as fp:
            fp.write(image_times)
        bucket.upload_file(path, path,
                           ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})
    
    
    def modify_img(self, hdul):
        """modifies and uploads the image"""
        hdul.verify('silentfix+warn')
        
        wave, t_rec = hdul[0].header['WAVELNTH'], hdul[0].header['T_OBS']
        data = hdul[0].data
        image_meta = str(wave), str(wave), t_rec, data.shape
        
        img_paths = Modify(data, image_meta).get_paths()
        
        return img_paths
    
    def upload_imgs(self, img_paths):
        """uploads all imgs in input to the s3 bucket"""
        for rtPath in img_paths:
            smallPath, bigPath, arcPath = self.make_thumb(rtPath)
            
            # Upload large File
            bucket.upload_file(rtPath, bigPath,
                               ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})
            
            # Upload Thumbnail
            bucket.upload_file(smallPath, smallPath,
                               ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})
            
            # Upload Archive
            if "orig" not in rtPath:
                bucket.upload_file(rtPath, arcPath,
                                   ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})
        
        return smallPath, bigPath, arcPath
    
    @staticmethod
    def make_thumb(rtPath):
        name = rtPath.split('/')[-1]
        arcPath = "renders/archive/" + "{}_{}".format(int(time()), name)
        smallPath = "renders/thumbs/" + name
        bigPath = 'renders/' + name
        makedirs("renders/thumbs/", exist_ok=True)
        
        imgDat = Image.open(rtPath)
        imgDat.thumbnail((512, 512))
        imgDat.save(smallPath)
        return smallPath, bigPath, arcPath
    
    def sleep_until_delay_elapsed(self, links=None):
        """ Make sure that the loop takes the right amount of time """
        self.wait_if_required(self.determine_delay(), links)
    

    def determine_delay(self):
        """ Determine how long to wait """
        delay = self.params.delay_seconds() + 0
        # return delay
        run_time_offset = time() - last_time
        delay -= run_time_offset
        delay = max(delay, 0)
        return delay
    
    def wait_if_required(self, delay, links=None):
        """ Wait if Required """
        
        # print("Waiting for {:0.0f} seconds ({} total)".format(delay, background_update_delay_seconds),
        #       flush=True, end='')
        # print('', end='', flush=True)
        # sys.stdout.flush()
        global picNum
        picNum = 0
        print("", flush=True)
        for ii in tqdm((range(int(delay))), desc="Waiting for {:0.0f} seconds".format(delay-1)):
            sleep(1)
            self.background_handler(ii, links, picNum)
        print("~~``~~\n")
    
    def background_handler(self, ii, links, picNum):
        """Change the desktop background every 60 seconds"""
        if set_local_background and not ii % 60:
            # print(abspath(links[picNum][1]))
            self.update_background(links[picNum][1])
            picNum += 1
            picNum = picNum % len(links)
    
    def update_background(self, local_path, test=False):
        """
        Update the System Background
    
        Parameters
        ----------
        local_path : str
            The local save location of the image
            :param test:
        """
        local_path = abspath(local_path)
        # print(local_path)
        assert isinstance(local_path, str)
        # print("Updating Background...", end='', flush=True)
        this_system = system()
        
        try:
            if this_system == "Windows":
                import ctypes
                SPI_SETDESKWALLPAPER = 0x14  # which command (20)
                SPIF_UPDATEINIFILE = 0x2  # forces instant update
                ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, local_path, SPIF_UPDATEINIFILE)
                # for ii in np.arange(100):
                #     ctypes.windll.user32.SystemParametersInfoW(19, 0, 'Fit', SPIF_UPDATEINIFILE)
            elif this_system == "Darwin":
                # from appscript import app, mactypes
                # try:
                #     app('Finder').desktop_picture.set(mactypes.File(local_path))
                # except Exception as e:
                #     if test:
                #         pass
                #     else:
                #         raise e
                print("Screw you, Macintosh, this don't work here")
                pass
            elif this_system == "Linux":
                import os
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-options 'scaled'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background primary-color 'black'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-uri {}".format(local_path))
            else:
                raise OSError("Operating System Not Supported")
            # print("Success")
        except Exception as e:
            print("Failed")
            raise e
        #
        # if self.params.is_debug():
        #     self.plot_stats()
        
        return 0
