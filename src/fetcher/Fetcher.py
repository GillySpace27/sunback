import os
print(os.getcwd())

from src.processor.Processor import Processor
import numpy as np

class Fetcher(Processor):
    """Gets some data"""
    filt_name = "Base Fetcher Class"
    description = "Use an Unnamed Fetcher"

    def __init__(self, params=None, quick=False, rp=None):
        # Initialize class variables
        super().__init__(params, quick, rp)
        # self.duration = ''
        self.frame_count = 0
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

    def cleanup(self):
        super().cleanup()
    # def process(self, params=None):
    #     self.fetch(params)


    def determine_image_path(self):
        if self.params.use_image_path():
            return self.params.use_image_path()
        else:
            if not os.path.exists(self.params.fits_directory()):
                return False
            # Parse File Paths
            all_paths = os.listdir(self.params.fits_directory())
            files = [x for x in all_paths if not os.path.isdir(os.path.join(self.params.fits_directory(), x))]
            fits_files = [x for x in files if "fits" in x]
            fits_dates = [x.split(".")[2] for x in fits_files]
            fits_dates_cleaned = [x.replace('-','/').replace("T", " ").replace("Z","") for x in fits_dates]
            times = [x.replace(":", "") for x in self.params.time_period()]

            # Test for Match
            correct = [times[0] <= x <= times[1] for x in fits_dates_cleaned]
            locs = np.where(correct)[0]

            if len(locs):
                # Do Stuff
                possible = [fits_files[x] for x in locs]
                wave = self.params.current_wave()
                while wave[0] == "0":
                    wave=wave[1:]
                right_wave = [wave in x for x in possible]
                loc2 = np.where(right_wave)[0]
                if len(loc2) == 1:
                    use_index = int(locs[loc2])
                elif len(loc2) > 1:
                    use_index = int(locs[loc2[self.frame_count]])
                    self.frame_count += 1
                else:
                    return False
            else:
                return False
                # raise FileNotFoundError(fits_dates_cleaned)

            use_file = fits_files[use_index]
            use_path = os.path.join(self.params.fits_directory(), use_file)
            # print("Img Use Path:{}".format(use_path))
        return self.params.use_image_path(use_path)

