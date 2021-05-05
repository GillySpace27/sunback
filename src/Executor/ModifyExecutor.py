from os import makedirs
from os.path import abspath, split
from platform import system
from time import sleep, time
from PIL import Image

from astropy.io import fits
from tqdm import tqdm

from Executor.Executor import Executor
from science.modify import Modify
from utils.file_util import discover_best_data_directory

# Flags
set_local_background = False

# Location of the Solar Images
archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"

# Initialization
last_time = time()
start_time = last_time

default_sleep = 30


# Web Version
class ModifyExecutor(Executor):
    def __init__(self, params):
        self.params = params
    
    def execute(self):
        """Loop over the wavelengths and normalize, then wait"""
        print("Processing Images...", flush=True)
        sleep(0.1)
        self.modify_img_series()
    
    def modify_img_series(self):
        """Processes the img series"""
        img_paths = []
        for path in tqdm(self.params.local_fits_paths()):
            with fits.open(path) as hdul:
                img_paths.extend(self.modify_img(hdul))
        self.params.local_img_paths(img_paths)
        print("Success!\n")
        sleep(1)
    
    def modify_img(self, hdul):
        """modifies and uploads the image"""
        hdul.verify('silentfix+warn')
        
        wave, t_rec = hdul[0].header['WAVELNTH'], hdul[0].header['T_OBS']
        data = hdul[0].data
        image_meta = str(wave), str(wave), t_rec, data.shape
        
        img_paths = Modify(data, image_meta).get_paths()
        self.make_thumbs(img_paths[0])
        return img_paths
    
    def make_thumbs(self, rtPath):
        smallPath, bigPath, arcPath = self.get_thumblinks(rtPath)
        imgDat = Image.open(rtPath)
        imgDat.thumbnail((512, 512))
        imgDat.save(smallPath)
        return smallPath, bigPath, arcPath
    
    @staticmethod
    def get_thumblinks(rtPath):
        name = split(rtPath)[-1]
        arcPath = "renders/archive/" + "{}_{}".format(int(time()), name)
        smallPath = "renders/thumbs/" + name
        bigPath = 'renders/' + name
        makedirs("renders/thumbs/", exist_ok=True)
        return smallPath, bigPath, arcPath
