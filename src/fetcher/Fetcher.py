
class Fetcher:
    """Gets some data"""
    
    def __init__(self):
        # Initialize class variables
        self.local_wave_directory = None
        self.image_folder = None
        self.movie_folder = None
        self.fits_folder = None
        self.fido_result = None
        self.fido_num = None
        
        self.local_fits_paths = []
        self.requested_files = []
        self.redownload = []
        self.file_size_mode = None
        self.temp_fits_pathbox = []
        self.waves_to_do = []
        
        self.start_time, self.start_time_long, self.start_string = '','',''
        self.end_time, self.end_time_long, self.end_time_string = '','',''
        self.params = None
    
    def fetch(self, url):
        raise NotImplementedError()
    

    
