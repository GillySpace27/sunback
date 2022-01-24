import os
print(os.getcwd())

from processor.Processor import Processor


class Fetcher(Processor):
    """Gets some data"""
    filt_name = "Base Fetcher Class"
    description = "Use an Unnamed Fetcher"
    
    def __init__(self, params=None, quick=False, rp=None):
        # Initialize class variables
        super().__init__(params, quick, rp)
        # self.load(params)
    
    def more_init(self):
        self.local_wave_directory = None
        self.image_folder = None
        self.movie_folder = None
        self.fits_folder = None
        self.fido_search_result = None
        self.fido_search_found_num = None
    
    def fetch(self, params=None):
        raise NotImplementedError()
    
    # def process(self, params=None):
    #     self.fetch(params)
 