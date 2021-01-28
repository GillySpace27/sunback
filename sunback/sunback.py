"""
sunback.py
A program that downloads recent png images from Gilly's S3 bucket and sets them to the desktop background

Handles the primary functions
"""

# Imports
from time import sleep, time
from os import getcwd, makedirs
from os.path import normpath, abspath, join, dirname, isdir
import astropy.units as u
start = time()
from threading import Barrier
bbb = Barrier(3, timeout=100)
from platform import system
import sys
import numpy as np

this_system = system()

if this_system == "Windows":
    # Windows Imports
    import sunpy.visualization.colormaps

elif this_system == "Linux":
    # Linux Imports
    import sunpy.visualization.colormaps

elif this_system == "Darwin":
    # Mac Imports
    pass

else:
    raise OSError("Operating System Not Supported")

# Main
debugg = True

print("Import took {:0.2f} seconds".format(time() - start))

class Parameters:
    """
    A container class for the run parameters of the program
    """
    seconds = 1
    minutes = 60 * seconds
    hours = 60 * minutes

    def __init__(self):
        """Sets all the attributes to None"""
        # Initialize Variables
        self.background_update_delay_seconds = None
        self.time_multiplier_for_long_display = None
        self.local_directory = None
        self.use_wavelengths = None
        self._resolution = 4096
        self.web_image_frame = None
        self.web_image_location = None
        self.web_paths = None
        self.file_ending = None
        self.run_time_offset = None
        self.time_file = None
        self.index_file = None
        self.debug_mode = False

        self.start_time = time()
        self.is_first_run = True
        self._do_HMI = True
        self._mode = 'all'
        self._do_mirror = False

        # Movie Defaults
        self._download_images = True
        self._overwrite_pngs = False
        self._make_compressed = False
        self._remove_old_images = False
        self._sonify_images = True
        self._sonify_limit = True
        self._do_171 = False
        self._do_304 = False
        self._something_changed = False
        self._allow_muxing = True

        self._stop_after_one = False

        self._time_period = None
        self._range_in_days = 5
        self._cadence = 10 * u.minute
        self._frames_per_second = 30
        self._bpm = 70

        self.set_default_values()

    def download_images(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._download_images = boolean
        if self._download_images:
            self.something_changed(True)
        return self._download_images

    def something_changed(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._something_changed = boolean
        return self._something_changed

    def overwrite_pngs(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._overwrite_pngs = boolean
        if self._overwrite_pngs:
            self.something_changed(True)
        return self._overwrite_pngs

    def make_compressed(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._make_compressed = boolean
        return self._make_compressed

    def remove_old_images(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._remove_old_images = boolean
        if self._remove_old_images:
            if self.something_changed():
                return True
        return False

    def sonify_images(self, boolean=None, mux=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._sonify_images = boolean
        if mux is not None:
            self.allow_muxing(mux)
        return self._sonify_images

    def allow_muxing(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._allow_muxing = boolean
        return self._allow_muxing

    def do_mirror(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._do_mirror = boolean
        return self._do_mirror

    def sonify_limit(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._sonify_limit = boolean
        return self._sonify_limit

    def do_171(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._do_171 = boolean
            if self._do_171:
                self.stop_after_one(True)
        return self._do_171

    def do_304(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._do_304 = boolean
            if self._do_304:
                self.stop_after_one(True)
        return self._do_304

    def stop_after_one(self, boolean=None):
        if boolean is not None:
            assert type(boolean) in [bool]
            self._stop_after_one = boolean
        return self._stop_after_one

    def range(self, days=None, hours=None):
        if days is not None or hours is not None:
            total_days = 0
            if days is not None:
                total_days += days
            if hours is not None:
                total_days += hours / 24
            self._range_in_days = total_days
        return self._range_in_days

    def cadence(self, cad=None):
        if cad is not None:
            self._cadence = cad * u.minute
        return self._cadence

    def time_period(self, period=None):
        if period is not None:
            self._time_period = period
        return self._time_period

    def frames_per_second(self, rate=None):
        if rate is not None:
            self._frames_per_second = rate
        return self._frames_per_second

    def bpm(self, bpm=None):
        if bpm is not None:
            self._bpm = bpm
        return self._bpm

    def check_real_number(self, number):
        assert type(number) in [float, int]
        assert number > 0

    def set_default_values(self):
        """Sets the Defaults for all the Parameters"""
        # SunbackMovie Parameters

        # Sunback Still Parameters
        #  Set Delay Time for Background Rotation
        self.set_delay_seconds(30 * self.seconds)
        self.set_time_multiplier(3)

        # Set File Paths
        self.set_local_directory()
        self.time_file = join(self.local_directory, 'time.txt')
        self.index_file = join(self.local_directory, 'index.txt')

        # Set Wavelengths
        self.set_wavelengths(['0171', '0193', '0211', '0304', '0131', '0335', '0094', 'HMIBC', 'HMIIF'])

        # Set Resolution
        self.set_download_resolution(2048)

        # Set Web Location
        self.set_web_image_frame("https://sdo.gsfc.nasa.gov/assets/img/latest/latest_{}_{}")

        # Add extra images
        new_web_path_1 = "https://sdo.gsfc.nasa.gov/assets/img/latest/f_211_193_171pfss_{}.jpg".format(self.resolution)
        self.append_to_web_paths(new_web_path_1, 'PFSS')

        # Select File Ending
        self.set_file_ending("{}_Now.png")

        return 0

    # Methods that Set Parameters
    def set_delay_seconds(self, delay):
        self.check_real_number(delay)
        self.background_update_delay_seconds = delay
        return 0

    def set_time_multiplier(self, multiplier):
        self.check_real_number(multiplier)
        self.time_multiplier_for_long_display = multiplier
        return 0

    def set_local_directory(self, path=None):
        if path is not None:
            self.local_directory = path
        else:
            self.local_directory = self.discover_best_default_directory()

        makedirs(self.local_directory, exist_ok=True)

    def set_wavelengths(self, waves):
        # [self.check_real_number(int(num)) for num in waves]
        self.use_wavelengths = waves
        self.use_wavelengths.sort()
        if self.has_all_necessary_data():
            self.make_web_paths()
        return 0

    def set_download_resolution(self, resolution):
        self.check_real_number(resolution)
        self._resolution = min([170, 256, 512, 1024, 2048, 3072, 4096], key=lambda x: np.abs(x - resolution))
        if self.has_all_necessary_data():
            self.make_web_paths()

    def resolution(self, resolution=None):
        if resolution is not None:
            self.check_real_number(resolution)
            self._resolution = min([170, 256, 512, 1024, 2048, 3072, 4096], key=lambda x: np.abs(x - resolution))
        return self._resolution

    def set_web_image_frame(self, path):
        self.web_image_frame = path
        if self.has_all_necessary_data():
            self.make_web_paths()

    def set_file_ending(self, string):
        self.file_ending = string

    # Methods that create something

    def make_web_paths(self):
        self.web_image_location = self.web_image_frame.format(self.resolution, "{}.jpg")
        self.web_paths = [self.web_image_location.format(wave) for wave in self.use_wavelengths]

    def append_to_web_paths(self, path, wave=' '):
        self.web_paths.append(path)
        self.use_wavelengths.append(wave)

    # Methods that return information or do something
    def has_all_necessary_data(self):
        if self.web_image_frame is not None:
            if self.use_wavelengths is not None:
                if self.resolution is not None:
                    return True
        return False

    def get_local_path(self, wave):
        return normpath(join(self.local_directory, self.file_ending.format(wave)))

    @staticmethod
    def discover_best_default_directory():
        """Determine where to store the images"""

        subdirectory_name = "sunback_images"
        if __file__ in globals():
            ddd = dirname(abspath(__file__))
        else:
            ddd = abspath(getcwd())

        while "dropbox".casefold() in ddd.casefold():
            ddd = abspath(join(ddd, ".."))

        directory = join(ddd, subdirectory_name)
        if not isdir(directory):
            makedirs(directory)

        # print("Image Location: {}".format(directory))
        # while not access(directory, W_OK):
        #     directory = directory.rsplit(sep)[0]
        #
        # print(directory)
        return directory

    def determine_delay(self):
        """ Determine how long to wait """

        delay = self.background_update_delay_seconds + 0
        # import pdb; pdb.set_trace()
        # if 'temp' in wave:
        #     delay *= self.time_multiplier_for_long_display

        self.run_time_offset = time() - self.start_time
        delay -= self.run_time_offset
        delay = max(delay, 0)
        return delay

    def wait_if_required(self, delay):
        """ Wait if Required """

        if delay <= 0:
            pass
        else:
            print("Waiting for {:0.0f} seconds ({} total)".format(delay, self.background_update_delay_seconds),
                  flush=True, end='')

            fps = 3
            for ii in (range(int(fps * delay))):
                sleep(1 / fps)
                print('.', end='', flush=True)
                # self.check_for_skip()
            print('Done')

    def sleep_until_delay_elapsed(self):
        """ Make sure that the loop takes the right amount of time """
        self.wait_if_required(self.determine_delay())

    def is_debug(self, debug=None):
        if debug is not None:
            self.debug_mode = debug
        return self.debug_mode

    def do_HMI(self, do=None):
        if do is not None:
            self._do_HMI = do
        return self._do_HMI

    def mode(self, mode=None):
        if mode is not None:
            self._mode = mode
        return self._mode


class Sunback:
    """
    The Primary Class that Does Everything

    Parameters
    ----------
    parameters : Parameters (optional)
        a class specifying run options
    """

    def __init__(self, parameters=None):
        """Initialize a new parameter object or use the provided one"""
        self.indexNow = 0
        if parameters:
            self.params = parameters
        else:
            self.params = Parameters()
        self.last_time = 0
        self.this_time = 1
        self.new_images = True
        self.fido_result = None
        self.fido_num = None
        self.renew_mask = True
        self.mask_num = [1, 2]

    # # Main Command Structure
    def start(self):
        """Select whether to run or to debug"""
        self.print_header()

        if self.params.is_debug():
            self.debug_mode()
        else:
            self.run_mode()

    def print_header(self):
        print("\nSunback: Live SDO Background Updater \nWritten by Chris R. Gilly")
        print("Check out my website: http://gilly.space\n")
        print("Delay: {} Seconds".format(self.params.background_update_delay_seconds))
        print("Coronagraph Mode: {} \n".format(self.params.mode()))

        if self.params.is_debug():
            print("DEBUG MODE\n")

    def debug_mode(self):
        """Run the program in a way that will break"""
        while True:
            self.execute_switch()

    def run_mode(self):
        """Run the program in a way that won't break"""

        fail_count = 0
        fail_max = 10

        while True:
            try:
                self.execute_switch()
            except (KeyboardInterrupt, SystemExit):
                print("\n\nOk, I'll Stop. Doot!\n")
                break
            except Exception as error:
                fail_count += 1
                if fail_count < fail_max:
                    print("I failed, but I'm ignoring it. Count: {}/{}\n\n".format(fail_count, fail_max))
                    continue
                else:
                    print("Too Many Failures, I Quit!")
                    sys.exit(1)

    def execute_switch(self):
        """Select which data source to draw from"""
        self.web_execute()
 
    def update_background(self, local_path, test=False):
        """
        Update the System Background

        Parameters
        ----------
        local_path : str
            The local save location of the image
            :param test:
        """
        local_path = abspath(local_path)
        # print(local_path)
        assert isinstance(local_path, str)
        print("Updating Background...", end='', flush=True)
        this_system = system()
        try:
            if this_system == "Windows":
                import ctypes
                SPI_SETDESKWALLPAPER = 0x14     #which command (20)
                SPIF_UPDATEINIFILE   = 0x2 #forces instant update
                ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, local_path, SPIF_UPDATEINIFILE)
                # for ii in np.arange(100):
                #     ctypes.windll.user32.SystemParametersInfoW(19, 0, 'Fit', SPIF_UPDATEINIFILE)
            elif this_system == "Darwin":
                from appscript import app, mactypes
                try:
                    app('Finder').desktop_picture.set(mactypes.File(local_path))
                except Exception as e:
                    if test:
                        pass
                    else:
                        raise e

            elif this_system == "Linux":
                import os
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-options 'scaled'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background primary-color 'black'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-uri {}".format(local_path))
            else:
                raise OSError("Operating System Not Supported")
            print("Success")
        except Exception as e:
            print("Failed")
            raise e
        #
        # if self.params.is_debug():
        #     self.plot_stats()

        return 0


    # Web Version
    
    def web_execute(self):
        self.web_get()
        self.web_run()

    def web_get(self):
        """Download the images if there are new ones"""
        self.download_all_objects_in_aws_folder()
        
    def web_run(self):
        """Loop over the wavelengths and normalize, set background, and wait"""

        for file_path in self.fileBox:
            print(file_path)
            self.params.start_time = time()
            self.name = file_path[-8:-4]
            if self.params.do_171() and "171" not in self.name:
                continue
            if self.params.do_304() and "304" not in self.name:
                continue

            print("Image: {}".format(self.name))

            # Update the Background
            self.update_background(file_path)

            # Wait for a bit
            self.params.sleep_until_delay_elapsed()

            if self.params.stop_after_one():
                sys.exit()

            print('')
            
    def download_all_objects_in_aws_folder(self):
        import boto3
        import os
        s3_resource = boto3.resource('s3')
        my_bucket = s3_resource.Bucket('gillyspace27-test-billboard')
        objects = my_bucket.objects.filter(Prefix='renders/')
        self.fileBox = []
        local_dir = self.params.discover_best_default_directory()
        print("Save Path: {}".format(local_dir))
        print("Downloading...")
        for obj in objects:
            path, filename = os.path.split(obj.key)
            if 'orig' in obj.key or 'archive' in obj.key or "thumbs" in obj.key or "4500" in obj.key:
                continue
            print('    ', filename)
            loc = join(local_dir, filename)
            my_bucket.download_file(obj.key, loc)
            self.fileBox.append(loc)
        print("All Downloads Complete\n\n")
        


# Helper Functions
def run(delay=60, debug=False, do171=False, do304=False):
    p = Parameters()
    p.set_delay_seconds(delay)
    p.do_171(do171)
    p.do_304(do304)

    if debug:
        p.is_debug(True)
        p.set_delay_seconds(10)
        p.do_HMI(False)

    Sunback(p).start()

def where():
    """Prints the location that the images are stored in."""
    p = Parameters()
    print(p.discover_best_default_directory())


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    where()
    run(debug=debugg)












# cdef int SINE = 0
# cdef int SINEIN = 17
# cdef int SINEOUT = 18
# cdef int COS = 1
# cdef int TRI = 2
# cdef int SAW = 3
# cdef int RSAW = 4
# cdef int HANN = 5
# cdef int HANNIN = 21
# cdef int HANNOUT = 22
# cdef int HAMM = 6
# cdef int BLACK = 7
# cdef int BLACKMAN = 7
# cdef int BART = 8
# cdef int BARTLETT = 8
# cdef int KAISER = 9
# cdef int SQUARE = 10
# cdef int RND = 11
# cdef int LINE = SAW
# cdef int PHASOR = SAW
# cdef int SINC = 23
# cdef int GAUSS = 24
# cdef int GAUSSIN = 25
# cdef int GAUSSOUT = 26
# cdef int PLUCKIN = 27
# cdef int PLUCKOUT = 28