import os
# print(os.getcwd())
import glob
from sunback.processor.Processor import Processor
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


    # def determine_image_path(self):
    #     if self.params.use_image_path():
    #         return self.params.use_image_path()
    #     else:
    #         if not os.path.exists(self.params.fits_directory()):
    #             return False
    #         # Parse File Paths
    #         all_paths = os.listdir(self.params.fits_directory())
    #         files = [x for x in all_paths if not os.path.isdir(os.path.join(self.params.fits_directory(), x))]
    #         fits_files = [x for x in files if "fits" in x]
    #         # try:
    #         #     fits_dates = [x.split(".")[2] for x in fits_files]
    #         # except IndexError:
    #         #     print("Error in fits_dates")
    #         #     return False
    #         # fits_dates_cleaned = [x.replace('-','/').replace("T", " ").replace("Z","") for x in fits_dates]

    #         # Correcting the extraction of dates and times from the filenames
    #         try:
    #             # Splitting filenames at the underscore character and extracting date and time parts
    #             fits_dates = [x[3:11] for x in fits_files]  # Extracting date directly using string slicing
    #             fits_times = [x.split("_")[1] for x in fits_files]
    #         except IndexError:
    #             print("Error in extracting dates and times")
    #             fits_dates_cleaned = []

    #         # Formatting the date and time strings
    #         fits_dates_cleaned = [f"{date[:4]}/{date[4:6]}/{date[6:]} {time[:2]}:{time[2:4]}:{time[4:]}"
    #                             for date, time in zip(fits_dates, fits_times)]


    #         times = [x.replace(":", "") for x in self.params.time_period()]

    #         # Test for Match
    #         correct = [times[0] <= x <= times[1] for x in fits_dates_cleaned]
    #         locs = np.where(correct)[0]

    #         if len(locs):
    #             # Do Stuff
    #             possible = [fits_files[x] for x in locs]
    #             wave = self.params.current_wave()
    #             while wave[0] == "0":
    #                 wave=wave[1:]
    #             right_wave = [wave in x for x in possible]
    #             loc2 = np.where(right_wave)[0]
    #             if len(loc2) == 1:
    #                 use_index = int(locs[loc2])
    #             elif len(loc2) > 1:
    #                 use_index = int(locs[loc2[self.frame_count]])
    #                 self.frame_count += 1
    #             else:
    #                 return False
    #         else:
    #             return False
    #             # raise FileNotFoundError(fits_dates_cleaned)

    #         use_file = fits_files[use_index]
    #         use_path = os.path.join(self.params.fits_directory(), use_file)
    #         # print("Img Use Path:{}".format(use_path))
    #     return self.params.use_image_path(use_path)

    def determine_image_path(self):
        # If the image path is defined in params, use it.
        if self.params.use_image_path():
            return self.params.use_image_path()

        # Check if the directory exists
        fits_dir = self.params.fits_directory()
        if not os.path.exists(fits_dir):
            return False
            raise FileNotFoundError(f"The directory {fits_dir} does not exist.")

        # Filter out only .fits files directly
        fits_files = glob.glob(os.path.join(fits_dir, "*.fits"))

        # # Extract dates and times
        # try:
        #     fits_dates = [x[3:11] for x in fits_files]
        #     fits_times = [x.split("_")[1] for x in fits_files]
        # except IndexError:
        #     raise ValueError("Error in extracting dates and times from filenames.")

        # # Format dates and times
        # formatted_dates_times = [
        #     f"{date[:4]}/{date[4:6]}/{date[6:]} {time[:2]}:{time[2:4]}:{time[4:]}"
        #     for date, time in zip(fits_dates, fits_times)
        # ]

        # start_time, end_time = [x.replace(":", "") for x in self.params.time_period()]

        # # Filter files based on time criteria
        # matching_files = [file for dt, file in zip(formatted_dates_times, fits_files)
        #                   if start_time <= dt <= end_time]

        matching_files = fits_files

        if not matching_files:
            raise FileNotFoundError("No matching files found in the given time period.")

        # Filter based on wave criteria
        wave = self.params.current_wave().lstrip('0')
        wave_matching_files = [file for file in matching_files if wave in file]

        if not wave_matching_files:
            raise FileNotFoundError(f"No files found matching wave {wave}.")

        # Determine the file to use
        if len(wave_matching_files) == 1:
            use_file = wave_matching_files[0]
        else:
            use_file = wave_matching_files[self.frame_count]
            self.frame_count += 1

        return os.path.join(fits_dir, use_file)