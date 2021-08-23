from astropy.io import fits
from tqdm import tqdm

from utils.file_util import save_fits_file


class Processor:
    def __init__(self, params):
        self.params = params
        self.done_paths = []
        self.skipped = 0
        self.name='Undefined'
        self.in_name = 'primary'
        self.out_name = 'filtered'
        
    def process(self):
        print(self.name+"...", flush=True, end='')
        self.modify_img_series()
        extra = '' if self.skipped == 0 else "But {} were skipped\n".format(self.skipped)
        print("Success! " + extra)
    
    def modify_img_series(self):
        """Processes the img series"""
        for fits_path in tqdm(self.params.local_fits_paths()):
            print(fits_path)
            with fits.open(fits_path) as hdul:
                hdul.verify('silentfix+warn')
                frame = self.modify_img(hdul, fits_path, self.in_name)
            save_fits_file(fits_path, hdul, frame, name= self.out_name)
            
    def modify_img(self, hdul, fits_path, original):
        raise NotImplementedError()