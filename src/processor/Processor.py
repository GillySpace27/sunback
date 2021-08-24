from astropy.io import fits
from tqdm import tqdm

from utils.file_util import save_frame_to_fits_file


class Processor:
    def __init__(self, params):
        self.params = params
        self.done_paths = []
        self.skipped = 0
        self.name='Undefined'
        self.in_name = 'primary'
        self.out_name = 'filtered'
        self.skipped = 0
        
    def process(self):
        print(self.name+"...", flush=True)
        self.modify_img_series()
        # extra = '' if self.skipped == 0 else "But {} were skipped\n".format(self.skipped)
        print("  Success!\n")# + extra)
    
    def modify_img_series(self):
        """Processes the img series"""
        for fits_path in tqdm(self.params.local_fits_paths(), desc="  "):
            frame = self.modify_img(fits_path, self.in_name)
            
    def modify_img(self, fits_path, original):
        raise NotImplementedError()