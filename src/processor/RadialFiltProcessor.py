from os import makedirs, listdir
from os.path import abspath, split, basename, dirname, join
from platform import system
from time import sleep, time
from PIL import Image
from astropy.nddata import block_reduce
import numpy as np
from astropy.io import fits
from tqdm import tqdm

# from fetcher.FidoFetcher im
# from executor.Executor import Executor
from processor.Processor import Processor
from science.modify import Modify
from utils.array_util import reduce_array
# from utils.file_util import discover_best_data_directory, find_done_paths, get_paths, load_fits_data, save_fits_file

# Flags
from utils.file_util import open_fits_hdul, save_frame_to_fits_file

set_local_background = False

# Location of the Solar Images
archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"

# Initialization
last_time = time()
start_time = last_time

default_sleep = 30

class RadialFiltProcessor(Processor):
    def __init__(self, p):
        self.name = '  Radial Filter'
        self.in_name = 'primary'
        self.out_name = 'SRN'
        self.params = p
        
    def modify_img(self, fits_path, in_name=None, out_name=None):
        if in_name: self.in_name = in_name
        if out_name: self.out_name = out_name
        
        frame = Modify(fits_path, self.in_name).get()
        save_frame_to_fits_file(fits_path, frame, field=self.out_name)
        
        return frame

    