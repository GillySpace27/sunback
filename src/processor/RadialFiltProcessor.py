from os import makedirs, listdir
from os.path import abspath, split, basename, dirname, join
from platform import system
from time import sleep, time
from PIL import Image
from astropy.nddata import block_reduce

from astropy.io import fits
from tqdm import tqdm

# from fetcher.FidoFetcher im
# from executor.Executor import Executor
from processor.Processor import Processor
from science.modify import Modify
from utils.array_util import reduce_array
from utils.file_util import discover_best_data_directory, find_done_paths, get_paths, load_fits_data, save_fits_file

# Flags
set_local_background = False

# Location of the Solar Images
archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"

# Initialization
last_time = time()
start_time = last_time

default_sleep = 30


# Web Version
class RadialFiltProcessor(Processor):
    def __init__(self, p):
        self.name = 'Radial Filter'
        self.in_name = 'primary'
        self.out_name = 'filtered'
        self.params = p
        
    def modify_img(self, hdul, fits_path, original):
        frame, wave, t_rec, center = load_fits_data(hdul, self.in_name)
        image_meta = str(wave), fits_path, t_rec, frame.shape
        frame = Modify(frame, image_meta, center=center).get()
        return frame

    

