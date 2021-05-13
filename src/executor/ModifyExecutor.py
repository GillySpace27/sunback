from os import makedirs, listdir
from os.path import abspath, split, basename, dirname
from platform import system
from time import sleep, time
from PIL import Image

from astropy.io import fits
from tqdm import tqdm

from executor.Executor import Executor
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
        self.done_paths = []
    
    def execute(self):
        """Loop over the wavelengths and normalize, then wait"""
        print("Processing Images...", flush=True)
        sleep(0.2)
        self.modify_img_series()
    
    def find_done_paths(self, full_path):
        """Find pngs that have already been made"""
        path = dirname(full_path)
        save_path = path.replace("fits", "png")
        self.done_paths = [x.casefold() for x in listdir(save_path)]
        return self.done_paths
    
    def modify_img_series(self):
        """Processes the img series"""
        img_paths = []
        skipped = 0
        for full_path in tqdm(self.params.local_fits_paths()):
            self.find_done_paths(full_path)
            name = basename(full_path).casefold().replace("fits", "png")
            if name in self.done_paths and not self.params.overwrite_pngs():
                one_path = full_path
            else:
                with fits.open(full_path) as hdul:
                    try:
                        one_path = self.modify_img(hdul, full_path)
                    except TypeError as e:
                        skipped += 1
                        print(e)
                        continue
            if type(one_path) not in [list]:
                one_path = [one_path]
            img_paths.extend(one_path)
            # break
        self.params.local_img_paths(img_paths)
        print("Success! But {} were skipped\n".format(skipped))
        sleep(1)
    
    def modify_img(self, hdul, path=None):
        """modifies and uploads the image"""
        hdul.verify('silentfix+warn')
        
        save_path = path.replace("fits", "png")
        filename = basename(save_path)
        if filename.casefold() in self.done_paths:
            if not self.params.overwrite_pngs():
                return save_path
        try:
            hh = 0
            wave, t_rec = hdul[hh].header['WAVELNTH'], hdul[hh].header['T_OBS']
        except:
            hh = 1
            wave, t_rec = hdul[hh].header['WAVELNTH'], hdul[hh].header['T_OBS']
            
        data = hdul[hh].data
        # image_meta = str(wave), str(wave), t_rec, data.shape
        image_meta = str(wave), save_path, t_rec, data.shape
        
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
