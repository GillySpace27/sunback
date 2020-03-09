"""
sunback.py
A program that downloads the most current images of the sun from the SDO satellite,
then sets each of the images to the desktop background in series.

Handles the primary functions
"""

# Imports
from time import localtime, timezone, strftime, sleep, time, struct_time
from urllib.request import urlretrieve
from os import getcwd, makedirs, rename, remove, listdir
from os.path import normpath, abspath, join, dirname, exists
from calendar import timegm
import astropy.units as u

start = time()
from sunpy.net import Fido, attrs as a
import sunpy.map
print("Import took {:0.2f} seconds".format(time()-start))

import platform
import sys
import numpy as np
import matplotlib as mpl
try:
    mpl.use('qt5agg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

this_system = platform.system()

if this_system == "Windows":
    #Windows Imports
    import sunpy.visualization.colormaps

elif this_system == "Linux":
    #Linux Imports
    import sunpy.visualization.colormaps

elif this_system == "Darwin":
    #Mac Imports
    pass

else:
    raise OSError("Operating System Not Supported")


# Main

debugg = True


def tr():
    import pdb;
    pdb.set_trace()


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
        self.resolution = None
        self.web_image_frame = None
        self.web_image_location = None
        self.web_paths = None
        self.file_ending = None
        self.run_time_offset = None
        self.time_file = None
        self.debug_mode = False

        self.start_time = time()
        self.is_first_run = True
        self._do_HMI = True
        self._mode = 'all'

        self.set_default_values()

    def check_real_number(self, number):
        assert type(number) in [float, int]
        assert number > 0

    def set_default_values(self):
        """Sets the Defaults for all the Parameters"""

        #  Set Delay Time for Background Rotation
        self.set_delay_seconds(30 * self.seconds)
        self.set_time_multiplier(3)

        # Set File Paths
        self.set_local_directory()
        self.time_file = join(self.local_directory, 'time.txt')


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
        self.resolution = min([170, 256, 512, 1024, 2048, 3072, 4096], key=lambda x: np.abs(x - resolution))
        if self.has_all_necessary_data():
            self.make_web_paths()

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
            directory = join(dirname(abspath(__file__)), subdirectory_name)
        else:
            directory = join(abspath(getcwd()), subdirectory_name)
        # print(directory)
        # while not access(directory, W_OK):
        #     directory = directory.rsplit(sep)[0]
        #
        # print(directory)

        return directory

    def determine_delay(self):
        """ Determine how long to wait """

        delay = self.background_update_delay_seconds + 0

        # if 'temp' in wave:
        #     delay *= self.time_multiplier_for_long_display

        self.run_time_offset = time() - self.start_time
        delay -= self.run_time_offset
        delay = max(delay, 0)
        return delay

    def wait_if_required(self, delay):
        """ Wait if Required """

        if self.is_first_run:
            self.is_first_run = False
        elif delay <= 0:
            pass
        else:
            # print("Took {:0.1f} seconds. ".format(self.run_time_offset), end='')
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
            self.debug_mode=debug
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

    def debug_mode(self):
        """Run the program in a way that will break"""
        while True:
            self.execute()

    def run_mode(self):
        """Run the program in a way that won't break"""

        fail_count = 0
        fail_max = 10

        while True:
            try:
                self.execute()
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

    def print_header(self):
        print("\nSunback: Live SDO Background Updater \nWritten by Chris R. Gilly")
        print("Check out my website: http://gilly.space\n")
        print("Delay: {} Seconds".format(self.params.background_update_delay_seconds))
        print("Coronagraph Mode: {} \n".format(self.params.mode()))

        if self.params.is_debug():
            print("DEBUG MODE\n")

    def execute(self):
        self.fido_search()

        while self.indexNow < self.fido_num:
            self.indexNow += 1
            self.run_AIA(self.indexNow - 1)
        if self.params.do_HMI():
            self.indexNow = 0
            self.run_HMI()
            self.run_HMI_white()
        self.indexNow = 0

    # # Main Loops

    def run_AIA(self, ii):
        """The Main Loop"""

        # Gather Data + Print
        this_name = self.fido_get_name_by_index(ii)

        # if '4500' in this_name:
        #     return
        # if this_name in ['0094', '0131'] and self.params.is_first_run:
        #     # print("Skip for Now\n")
        #     return
        #
        # # if int(this_name) < 1000:
        # #     return
        #
        # if self.params.is_debug():
        #     if this_name in ['1600', '1700']:
        #         return

        if this_name not in ['304']:
            return

        print("Image: {}".format(this_name))

        # Retrieve and Prepare the Image
        image_path = self.make_image(ii)

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        self.update_background(image_path)

        print('')

    def run_AIA_movie(self, ii):
        """The Main Loop"""

        # Gather Data + Print
        this_name = self.fido_get_name_by_index(ii)

        # if '4500' in this_name:
        #     return
        # if this_name in ['0094', '0131'] and self.params.is_first_run:
        #     # print("Skip for Now\n")
        #     return
        #
        # # if int(this_name) < 1000:
        # #     return
        #
        # if self.params.is_debug():
        #     if this_name in ['1600', '1700']:
        #         return
        #
        # # if this_name not in ['211']:
        # #     return

        if this_name not in ['304']:
            return

        print("Image: {}".format(this_name))

        # Retrieve and Prepare the Image
        # image_path = self.make_image(ii)
        image_path = self.make_movie(ii)

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        # self.update_background(image_path)



        print('')

    def run_HMI(self):
        """The Secondary Loop"""

        web_path = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIBC.jpg"

        self.params.start_time = time()

        print("Image: {}".format('HMIBC'))
        # Define the Image
        self.hmi_path = normpath(join(self.params.local_directory, "HMIBC_Now.jpg"))

        # Download the Image
        self.download_image(self.hmi_path, web_path)

        # Modify the Image
        print("Modifying Image...", end="")
        new_path = self.plot_and_save(mpimg.imread(self.hmi_path), ('HMI', self.hmi_path, "Magnetic Field", -1), None)

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        self.update_background(new_path)

        print('')

    def run_HMI_white(self):
        """The Secondary Loop"""

        web_path = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIIF.jpg"

        self.params.start_time = time()

        print("Image: {}".format('HMIF'))
        # Define the Image
        self.hmi_path = normpath(join(self.params.local_directory, "HMIF_Now.jpg"))

        # Download the Image
        self.download_image(self.hmi_path, web_path)

        # Modify the Image
        print("Modifying Image...", end="")
        new_path = self.plot_and_save(mpimg.imread(self.hmi_path), ('HMI', self.hmi_path, "White Light", -1))

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        self.update_background(new_path)

        print('')

    def download_image(self, local_path, web_path):
        """
        Download an image and save it to file

        Go to the internet and download an image

        Parameters
        ----------
        web_path : str
            The web location of the image

        local_path : str
            The local save location of the image
        """
        tries = 3

        for ii in range(tries):
            try:
                print("Downloading Image...", end='', flush=True)
                urlretrieve(web_path, local_path)
                print("Success", flush=True)
                return 0
            except KeyboardInterrupt:
                raise
            except Exception as exp:
                print("Failed {} Time(s).".format(ii + 1), flush=True)
                if ii == tries - 1:
                    raise exp

    # # Level 2 ##

    def fido_search(self):
        """ Find the Most Recent Images """
        self.renew_mask = True
        self.fido_num = 0
        tries = 0
        minute_range = 18

        max_tries = 20 if self.params.is_debug() else 10
        min_num = 3
        max_num = 13

        while True:
            if tries > max_tries:
                self.new_images = False
                break
            try:
                # Define Time Range
                fmt_str = '%Y/%m/%d %H:%M'
                early = strftime(fmt_str, localtime(time() - minute_range * 60 + timezone))
                now = strftime(fmt_str, localtime(time() + timezone))

                # Find Results
                self.fido_result = Fido.search(a.Time(early, now), a.Instrument('aia'))
                self.fido_num = self.fido_result.file_num
                if self.params.is_debug():
                    print(self.fido_num, '\t', minute_range)
                # Change time range if wrong number of records
                if self.fido_num > max_num:
                    # tries += 1
                    minute_range -= 2
                    if tries > 3:
                        if (tries - max_num) < 30:
                            continue
                        minute_range -= 4
                    continue
                if self.fido_num < min_num:
                    tries += 1
                    minute_range += 2
                    if tries > 3:
                        minute_range += 10
                    if tries > 7:
                        minute_range += 30
                    continue

                self.this_time = int(self.fido_result.get_response(0)[0].time.start)
                self.new_images = self.last_time < self.this_time
                break
            except ConnectionError:
                self.new_images = False
                break

        if self.new_images:
            print("Search Found {} new images at {}\n".format(self.fido_num,
                                                              self.parse_time_string(str(self.this_time), 2)), flush=True)
            self.last_time = self.this_time

            with open(self.params.time_file, 'w') as fp:
                fp.write(str(self.this_time) + '\n')
                fp.write(str(self.fido_num) + '\n')
                fp.write(str(self.fido_result.get_response(0)))
        else:
            print("No New Images, using Cached Data\n")
            self.fido_result = []

            with open(self.params.time_file, 'r') as fp:
                self.this_time = int(fp.readline())
                self.fido_num = int(fp.readline())
                fp.readline()
                fp.readline()
                fp.readline()
                while True:
                    line = fp.readline()
                    self.fido_result.append(line)
                    if len(line) == 0:
                        break
                # print(self.fido_result)
# fp.write(str(self.this_time) + '\n')
                # fp.write(str(self.fido_num) + '\n')
                # fp.write(str(self.fido_result.get_response(0)))



    def fido_get_name_by_index(self, ind):
        self.params.start_time = time()
        if self.new_images:
            name = self.fido_result[0, ind].get_response(0)[0].wave.wavemin
        else:
            name = self.fido_result[ind][-6:-2]
            if name[-1] == '.':
                name = name[:-1]
            if name[0] == ' ':
                name = name[1:]
        while len(name) < 4:
            name = '0' + name
        return name

    def make_movie(self, ii):
        # Download the fits data
        image_data = self.fido_download_by_index(ii)

        # Generate a png image
        image_path = self.fits_to_image(image_data)

        return image_path


    def make_image(self, ii):
        # Download the fits data
        image_data = self.fido_download_by_index(ii)

        # Generate a png image
        image_path = self.fits_to_image(image_data)

        return image_path

    # Level 3

    def fido_download_by_index(self, ind):
        """Retrieve a result by index and save it to file"""

        tries = 3

        if self.new_images:
            for ii in range(tries):
                try:
                    print("Downloading Fits Data...", end='', flush=True)
                    result = self.fido_retrieve_result(self.fido_result[0, ind])
                    print("Success", flush=True)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exp:
                    print("Failed {} Time(s).".format(ii + 1), flush=True)
                    if ii == tries-1:
                        return self.use_cached(ind)
        else:
            result = self.use_cached(ind)

        out = [x for x in result]
        out.append(ind)
        return out

    def fido_retrieve_result(self, this_result):
        """Retrieve a result and save it to file"""
        # Make the File Name
        name, save_path = self.get_paths(this_result)

        # Download and Rename the File
        time_string = self.fido_download(this_result, save_path)

        return name, save_path, time_string

    def fido_download(self, this_result, save_path):
        original = sys.stderr
        sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w')
        downloaded_files = Fido.fetch(this_result, path=self.params.local_directory)
        sys.stderr = original

        if exists(save_path):
            remove(save_path)
        rename(downloaded_files[0], save_path)

        try:
            time_string = self.parse_time_string(downloaded_files)
        except:
            time_string = "xxxx"

        return time_string

    def use_cached(self, ind):
        print("Using Cached Data...", end='', flush=True)
        result = self.list_files1(self.params.local_directory, 'fits')
        file_name = [x for x in result][ind]
        full_name = file_name[:4]

        with open(self.params.time_file, 'r') as fp:
            time_stamp = fp.read()
        time_string = self.parse_time_string(str(time_stamp), 2)
        save_path = join(self.params.local_directory, file_name)
        print("Success", flush=True)
        return full_name, save_path, time_string

    def fits_to_image(self, image_data):
        """Modify the Fits image into a nice png"""
        print("Generating Image...", flush=True)

        originalData, image_data = self.load_fits(image_data)

        # Modify the data
        data = self.image_modify(originalData)

        # Plot the Data
        new_path = self.plot_and_save(data, image_data, originalData)

        return new_path

    def load_fits(self, image_data):
        # Load the Fits File from disk
        full_name, save_path, time_string, ii = image_data

        for ind in np.arange(4):
            try:
                # Parse Inputs
                full_name, save_path, time_string, ii = image_data
                my_map = sunpy.map.Map(save_path)
                break
            except TypeError as e:
                if ind < 3:
                    image_data = self.fido_download_by_index(ii)
                else:
                    raise e
        data = my_map.data
        return data, image_data

    def image_modify(self, data):
        data = data + 0
        self.radial_analyze(data, False)
        data = self.absqrt(data)
        data = self.coronagraph(data)
        data = self.vignette(data)

        return data

    def plot_and_save(self, data, image_data, original_data=None):
        full_name, save_path, time_string, ii = image_data
        name = self.clean_name_string(full_name)

        for processed in [False, True]:

            if not self.params.is_debug():
                if not processed:
                    continue
            if not processed:
                if original_data is None:
                    continue

            # Create the Figure
            fig, ax = plt.subplots()
            self.blankAxis(ax)

            inches = 10
            fig.set_size_inches((inches,inches))

            pixels = data.shape[0]
            dpi = pixels / inches


            if 'hmi' in name.casefold():
                inst = ""
                plt.imshow(data, origin='upper', interpolation=None)
                # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                plt.tight_layout(pad=5.5)
                height = 1.05

            else:
                inst = '  AIA'
                plt.imshow(data if processed else self.normalize(original_data), cmap='sdoaia{}'.format(name), origin='lower', interpolation=None,
                           vmin=self.vmin_plot, vmax=self.vmax_plot)
                plt.tight_layout(pad=0)
                height = 0.95

            # Annotate with Text
            buffer = '' if len(name) == 3 else '  '
            buffer2 ='    ' if len(name) == 2 else ''


            title = "{}    {} {}, {}{}".format(buffer2, inst, name, time_string, buffer)
            title2 = "{} {}, {}".format(inst, name, time_string)
            ax.annotate(title, (0.125, height + 0.02), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')
            # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
            the_time = strftime("%I:%M%p").lower()
            if the_time[0] == '0':
                the_time = the_time[1:]
            ax.annotate(the_time, (0.125, height), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')




            # Format the Plot and Save
            self.blankAxis(ax)
            middle = '' if processed else "_orig"
            new_path = save_path[:-5]+middle+".png"


            try:
                plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
                print("\tSaved {} Image".format('Processed' if processed else "Unprocessed"))
            except PermissionError:
                new_path = save_path[:-5]+"_b.png"
                plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
                print("Success")
            except Exception as e:
                print("Failed...using Cached")
                if self.params.is_debug():
                    raise e
            plt.close(fig)

        return new_path

    def update_background(self, local_path, test=False):
        """
        Update the System Background

        Parameters
        ----------
        local_path : str
            The local save location of the image
        """
        print("Updating Background...", end='', flush=True)
        assert isinstance(local_path, str)
        local_path = normpath(local_path)

        this_system = platform.system()

        try:
            if this_system == "Windows":
                import ctypes
                ctypes.windll.user32.SystemParametersInfoW(20, 0, local_path, 0)
                # ctypes.windll.user32.SystemParametersInfoW(19, 0, 'Fill', 0)

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

        if self.params.is_debug():
            self.plot_stats()

        return 0


    # Level 4

    @staticmethod
    def list_files1(directory, extension):
        from os import listdir
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    def get_paths(self, this_result):
        self.name = this_result.get_response(0)[0].wave.wavemin
        while len(self.name)<4:
            self.name = '0'+ self.name
        file_name = '{}_Now.fits'.format(self.name)
        save_path = join(self.params.local_directory, file_name)
        return self.name, save_path

    @staticmethod
    def parse_time_string(downloaded_files, which=0):
        if which == 0:
            time_string = downloaded_files[0][-25:-10]
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = int(time_string[9:11])
            minute = time_string[11:13]
        else:
            time_string = downloaded_files
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = int(time_string[8:10])
            minute = time_string[10:12]

        hour = str(hour_raw%12)
        if hour == '0':
            hour = 12
        suffix = 'pm' if hour_raw > 12 else 'am'
        from time import mktime
        struct_time = (int(year), int(month), int(day), hour_raw, int(minute), 0, 0, 0, -1)

        new_time_string = strftime("%I:%M%p %m/%d/%Y ", localtime(timegm(struct_time))).lower()
        if new_time_string[0] == '0':
            new_time_string = new_time_string[1:]

        # print(year, month, day, hour, minute)
        # new_time_string = "{}:{}{} {}/{}/{} ".format(hour, minute, suffix, month, day, year)
        return new_time_string

    @staticmethod
    def clean_name_string(full_name):
        # Make the name strings
        name = full_name + ''
        while name[0] == '0':
            name = name[1:]
        return name

    @staticmethod
    def blankAxis(ax):
        ax.patch.set_alpha(0)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='none', which='both',
                       top=False, bottom=False, left=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')


    # Data Manipulations

    @staticmethod
    def reject_outliers(data):
        # # Reject Outliers
        # a = data.flatten()
        # remove_num = 20
        # ind = argpartition(a, -remove_num)[-remove_num:]
        # a[ind] = nanmean(a)*4
        # data = a.reshape(data.shape)

        data[data>10] = np.nan

        return data

    @staticmethod
    def absqrt(data):
        return np.sqrt(np.abs(data))

    @staticmethod
    def normalize(data):
        high = 98
        low = 15

        lowP = np.nanpercentile(data, low)
        highP = np.nanpercentile(data, high)
        return (data - lowP) / (highP - lowP)

    def vignette(self, data):

        mask = self.radius > (self.noise_radii)
        data[mask] = np.nan
        return data

    def vignette2(self, data):

        mask = np.isclose(self.radius, self.tRadius, atol = 2)
        data[mask] = 1

        mask = np.isclose(self.radius, self.noise_radii, atol = 2)
        data[mask] = 1
        return data

    def coronagraph(self, data):
        original = sys.stderr
        sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')

        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        dat_corona = (self.dat_flat - self.fakeMin[radius_bin]) / \
                         (self.fakeMax[radius_bin] - self.fakeMin[radius_bin])

        sys.stderr = original

        # Deal with too hot things
        self.vmax = 1
        self.vmax_plot = 1.8

        hot = dat_corona>self.vmax
        dat_corona[hot] = dat_corona[hot]**(1/2)

        # Deal with too cold things
        self.vmin = 0.06
        self.vmin_plot = -0.03

        cold = dat_corona < self.vmin
        dat_corona[cold] = -((np.abs(dat_corona[cold]-self.vmin)+1)**(1/2)-1)+self.vmin

        self.dat_coronagraph = dat_corona
        dat_corona_square = dat_corona.reshape(data.shape)

        if self.renew_mask or self.params.mode() == 'r':
            self.corona_mask = self.get_mask(data)
            self.renew_mask = False

        data = self.normalize(data)

        data[self.corona_mask] = dat_corona_square[self.corona_mask]

        # inds = np.argsort(self.rad_flat)
        # rad_sort = self.rad_flat[inds]
        # dat_sort = dat_corona[inds]
        #
        # plt.figure()
        # # plt.yscale('log')
        # plt.scatter(rad_sort[::30], dat_sort[::30], c='k')
        # plt.show()

        return data

    def get_mask(self, dat_out):

        corona_mask = np.full_like(dat_out, False, dtype=bool)
        rezz = corona_mask.shape[0]
        half = int(rezz / 2)

        mode = self.params.mode()



        if type(mode) in [float, int]:
            mask_num = mode
        elif 'y' in mode:
            mask_num = 1
        elif 'n' in mode:
            mask_num = 2
        else:
            if 'r' in mode:
                if len(mode) < 2:
                    mode += 'a'

            if 'a' in mode:
                top = 8
                btm = 1
            elif 'h' in mode:
                top = 6
                btm = 3
            elif 'd' in mode:
                top = 8
                btm = 7
            elif 'w' in mode:
                top = 2
                btm = 1
            else:
                print('Unrecognized Mode')
                top = 8
                btm = 1

            ii = 0
            while True:
                mask_num = np.random.randint(btm, top+1)
                if mask_num not in self.mask_num:
                    self.mask_num.append(mask_num)
                    break
                ii += 1
                if ii > 10:
                    self.mask_num = []


        if mask_num == 1:
            corona_mask[:, :] = True

        if mask_num == 2:
            corona_mask[:, :] = False

        if mask_num == 3:
            corona_mask[half:, :] = True

        if mask_num == 4:
            corona_mask[:half, :] = True

        if mask_num == 5:
            corona_mask[:, half:] = True

        if mask_num == 6:
            corona_mask[:, :half] = True

        if mask_num == 7:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True

        if mask_num == 8:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True
            corona_mask = np.invert(corona_mask)

        return corona_mask

    # Basic Analysis

    def radial_analyze(self, data, plotStats=False):
        self.offset = np.abs(np.min(data))

        data += self.offset

        self.make_radius(data)
        self.sort_radially(data)
        self.bin_radially()
        self.fit_curves()

        if plotStats:
            self.plot_stats()

    def make_radius(self, data):


        self.rez = data.shape[0]
        centerPt = self.rez / 2
        xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
        xc, yc = xx - centerPt, yy - centerPt

        self.extra_rez = 4

        self.sRadius = 400 * self.extra_rez
        self.tRadius = self.sRadius * 1.28
        self.radius = np.sqrt(xc * xc + yc * yc) * self.extra_rez
        self.rez*= self.extra_rez

    def sort_radially(self, data):
        # Create arrays sorted by radius
        self.rad_flat = self.radius.flatten()
        self.dat_flat = data.flatten()
        inds = np.argsort(self.rad_flat)
        self.rad_sort = self.rad_flat[inds]
        self.dat_sort = self.dat_flat[inds]

    def bin_radially(self):

        # Bin the intensities by radius
        self.radBins = [[] for x in np.arange(self.rez)]
        binInds = np.asarray(np.floor(self.rad_sort), dtype=np.int32)
        for ii, binI in enumerate(binInds):
            self.radBins[binI].append(self.dat_sort[ii])

        # Find the statistics by bin
        self.binMax = np.zeros(self.rez)
        self.binMin = np.zeros(self.rez)
        self.binMean = np.zeros(self.rez)
        self.binMed = np.zeros(self.rez)

        for ii, it in enumerate(self.radBins):
            item = np.asarray(it)

            idx = np.isfinite(item)
            finite = item[idx]
            idx2 = np.nonzero(finite-self.offset)
            subItems = finite[idx2]

            if len(subItems)>0:
                self.binMax[ii] = np.percentile(subItems, 97) #np.nanmax(subItems)
                self.binMin[ii] = np.percentile(subItems, 2) #np.min(subItems)
                self.binMean[ii] = np.mean(subItems)
                self.binMed[ii] = np.median(subItems)
            else:
                self.binMax[ii] = np.nan
                self.binMin[ii] = np.nan
                self.binMean[ii] = np.nan
                self.binMed[ii] = np.nan

    def fit_curves(self):
        # Input Stuff
        self.radAbss = np.arange(self.rez)
        self.highCut = 730 * self.extra_rez
        theMin = 380 * self.extra_rez
        near_limb = np.arange(theMin, theMin + 50 * self.extra_rez)

        # Find the derivative of the binned mean
        self.diff_mean = np.diff(self.binMean)
        self.diff_mean += np.abs(np.nanmin(self.diff_mean))
        self.diff_mean /= np.nanmean(self.diff_mean) / 100

        # Locate the Limb
        self.limb_radii = np.argmin(self.diff_mean[near_limb]) + theMin
        self.lCut = self.limb_radii - 10 * self.extra_rez
        self.hCut = self.limb_radii + 10 * self.extra_rez


        # Split into three regions
        self.low_abs = self.radAbss[:self.lCut]
        self.low_max = self.binMax[:self.lCut]
        self.low_min = self.binMin[:self.lCut]

        self.mid_abs = self.radAbss[self.lCut:self.hCut]
        self.mid_max = self.binMax[self.lCut:self.hCut]
        self.mid_min = self.binMin[self.lCut:self.hCut]

        self.high_abs = self.radAbss[self.hCut:]
        self.high_max = self.binMax[self.hCut:]
        self.high_min = self.binMin[self.hCut:]

        # Filter the regions separately
        from scipy.signal import savgol_filter

        lWindow = 20 * self.extra_rez + 1
        mWindow = 4 * self.extra_rez + 1
        hWindow = 30 * self.extra_rez + 1
        fWindow = int(3 * self.extra_rez) + 1

        rank = 3

        low_max_filt = savgol_filter(self.low_max, lWindow, rank)

        mid_max_filt = savgol_filter(self.mid_max, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)

        high_max_filt = savgol_filter(self.high_max, hWindow, rank)

        low_min_filt = savgol_filter(self.low_min, lWindow, rank)
        mid_min_filt =  savgol_filter(self.mid_min, mWindow, rank)
        high_min_filt = savgol_filter(self.high_min, hWindow, rank)

        # Fit the low curves
        p = np.polyfit(self.low_abs, self.fill_start(low_max_filt), 9)
        low_max_fit = np.polyval(p, self.low_abs)
        p = np.polyfit(self.low_abs, self.fill_start(low_min_filt), 9)
        low_min_fit = np.polyval(p, self.low_abs)


        # Build output curves
        self.fakeAbss = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.fakeMax = np.hstack((low_max_fit, mid_max_filt, high_max_filt))
        self.fakeMin = np.hstack((low_min_fit, mid_min_filt, high_min_filt))

        # Filter again to smooth boundaraies
        self.fakeMax = self.fill_end(self.fill_start(savgol_filter(self.fakeMax, fWindow, rank)))
        self.fakeMin = self.fill_end(self.fill_start(savgol_filter(self.fakeMin, fWindow, rank)))

        # Locate the Noise Floor
        noiseMin = 550 * self.extra_rez - self.hCut
        near_noise = np.arange(noiseMin, noiseMin + 100 * self.extra_rez)
        self.diff_max_abs = self.high_abs[near_noise]
        self.diff_max = np.diff(high_max_filt)[near_noise]
        self.diff_max += np.abs(np.nanmin(self.diff_max))
        self.diff_max /= np.nanmean(self.diff_max) / 100
        self.noise_radii = np.argmin(self.diff_max) + noiseMin + self.hCut
        self.noise_radii = 565 * self.extra_rez

    def fill_end(self, use):
        iii = -1
        val = use[iii]
        while np.isnan(val):
            iii -= 1
            val = use[iii]
        use[iii:] = val
        return use

    def fill_start(self, use):
        iii = 0
        val = use[iii]
        while np.isnan(val):
            iii += 1
            val = use[iii]
        use[:iii] = val
        return use

    def plot_stats(self):

        fig, (ax0, ax1) = plt.subplots(2, 1, True)
        ax0.scatter(self.n2r(self.rad_sort[::30]), self.dat_sort[::30], c='k', s=2)
        ax0.axvline(self.n2r(self.limb_radii), ls='--', label="Limb")
        ax0.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        ax0.axvline(self.n2r(self.lCut), ls=':')
        ax0.axvline(self.n2r(self.hCut), ls=':')
        # ax0.axvline(self.tRadius, c='r')
        ax0.axvline(self.n2r(self.highCut))


        # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
        # plt.plot(self.radAbss[:-1] + 0.5, self.diff_mean, 'r:')

        ax0.plot(self.n2r(self.low_abs), self.low_max, 'm', label="Percentile")
        ax0.plot(self.n2r(self.low_abs), self.low_min, 'm')
        # plt.plot(self.low_abs, self.low_max_fit, 'r')
        # plt.plot(self.low_abs, self.low_min_fit, 'r')

        ax0.plot(self.n2r(self.high_abs), self.high_max, 'c', label="Percentile")
        ax0.plot(self.n2r(self.high_abs), self.high_min, 'c')

        ax0.plot(self.n2r(self.mid_abs), self.mid_max, 'y', label="Percentile")
        ax0.plot(self.n2r(self.mid_abs), self.mid_min, 'y')
        # plt.plot(self.high_abs, self.high_min_fit, 'r')
        # plt.plot(self.high_abs, self.high_max_fit, 'r')

        try:
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMin, 'g')
        except:
            ax0.plot(self.n2r(self.radAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.radAbss), self.fakeMin, 'g')

        # plt.plot(radAbss, binMax, 'c')
        # plt.plot(self.radAbss, self.binMin, 'm')
        # plt.plot(self.radAbss, self.binMean, 'y')
        # plt.plot(radAbss, binMed, 'r')
        # plt.plot(self.radAbss, self.binMax, 'b')
        # plt.plot(radAbss, fakeMin, 'r')
        # plt.ylim((-100, 10**3))
        # plt.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        ax0.set_xlim((0,self.n2r(self.highCut)))
        ax0.legend()
        fig.set_size_inches((8,12))
        ax0.set_yscale('log')

        ax1.scatter(self.n2r(self.rad_flat[::10]), self.dat_coronagraph[::10], c='k', s=2)
        ax1.set_ylim((-0.25,2))

        ax1.axhline(self.vmax, c='r', label='Confinement')
        ax1.axhline(self.vmin, c='r')
        ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
        ax1.axhline(self.vmin_plot, c='orange')

        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))

        ax1.legend()
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax0.set_ylabel(r"Absolute Intensity (Counts)")

        plt.tight_layout()
        if self.params.is_debug():
            file_name = '{}_Radial.png'.format(self.name)
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)

            file_name = '{}_Radial_zoom.png'.format(self.name)
            ax0.set_xlim((0.9,1.1))
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)
            # plt.show()
            plt.close(fig)
        else:
            plt.show()

    def n2r(self, n):
        if True:
            return n / self.limb_radii
        else:
            return n


class SunbackMovie:
    """
    The Primary Class that Does Everything

    Parameters
    ----------
    parameters : Parameters (optional)
        a class specifying run options
    """

    def __init__(self, parameters=None):
        """Initialize a new parameter object or use the provided one"""
        self.fits_analysis_done = False
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
        self.wavelengths = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700']
        self.waveNum = len(self.wavelengths)
        self.this_name = None


    # # Main Command Structure
    def start(self):
        """Select whether to run or to debug"""
        self.print_header()

        if self.params.is_debug():
            self.debug_mode()
        else:
            self.run_mode()

    def debug_mode(self):
        """Run the program in a way that will break"""
        while True:
            self.execute()

    def run_mode(self):
        """Run the program in a way that won't break"""

        fail_count = 0
        fail_max = 10

        while True:
            try:
                self.execute()
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

    def print_header(self):
        print("\nSunback: Live SDO Background Updater \nWritten by Chris R. Gilly")
        print("Check out my website: http://gilly.space\n")
        print("Delay: {} Seconds".format(self.params.background_update_delay_seconds))
        print("Coronagraph Mode: {} \n".format(self.params.mode()))

        if self.params.is_debug():
            print("DEBUG MODE\n")

    def execute(self):
        while self.indexNow < self.waveNum:
            self.indexNow += 1
            self.run_AIA_movie(self.indexNow - 1)
        self.indexNow = 0

    # # Main Loops

    def skip(self):

        # Skip Logic
        # if '4500' in self.this_name:
        #     return
        if self.this_name in ['0094', '0131', '0171'] and self.params.is_first_run:
            # print("Skip for Now\n")
            return 1
        #
        # if self.params.is_debug():
        #     if self.this_name in ['1600', '1700']:
        #         return
        #
        # # if int(self.this_name) < 1000:
        # #     return

        # if '171' not in self.this_name:
        #     return 1

        return 0


    def run_AIA_movie(self, ii):
        """The Main Loop"""

        self.download_images = True
        self.overwrite_pngs = False
        self.make_uncompressed = False
        self.remove_old_images = False

        self.range_in_hours = 7*24
        self.cadence = 6*u.minute
        self.frame_rate = 30




        # Gather Data + Print
        self.this_name = self.wavelengths[ii]


        if self.skip():
            return

        # Define Time Range
        self.get_time_range(self.range_in_hours)



        # Find the Files
        print("Movie: {}".format(self.this_name))

        # Download the fits data
        if self.download_images:
            self.fido_download_fits_series()

        # Analyze the Dataset
        # self.fits_analyze()

        # Generate the png images
        self.fits_to_pngs()

        # Generate the Movie
        self.pngs_to_movie()

        print("Wavlength Complete: {}".format(self.this_name))
        sys.exit()


    def get_time_range(self, range_in_hours):
        """Define Time Range, on the hour"""
        fmt_str = '%Y/%m/%d %H:%M'
        thisEarly = list(localtime(time() - range_in_hours * 60 * 60 + timezone))
        thisEarly[4] = 0
        thisEarly[5] = 0
        earlyStruct = struct_time(thisEarly)
        self.early = strftime(fmt_str, earlyStruct)
        self.earlyLong = int(strftime('%Y%m%d%H%M%S', earlyStruct))
        self.earlyString = self.parse_time_string(str(self.earlyLong), 2)

        thisNow = list(localtime(time() + timezone))
        thisNow[4] = 0
        thisNow[5] = 0
        nowStruct = struct_time(thisNow)
        self.now = strftime(fmt_str, nowStruct)
        self.nowLong = int(strftime('%Y%m%d%H%M%S', nowStruct))
        self.nowString = self.parse_time_string(str(self.nowLong), 2)

    def fido_download_fits_series(self):
        """ Find the Most Recent Images """
        print("Downloading Fits Data between {} and {}...".format(self.earlyString, self.nowString), flush=True)

        # Find Results
        self.fido_result = Fido.search(a.Time(self.early, self.now), a.Instrument('aia'), a.Wavelength(int(self.this_name)*u.angstrom), a.vso.Sample(self.cadence), a.Level(1.5))

        # Analyze Results
        self.import_fido_information()

        # Download Results
        downloaded_files = Fido.fetch(self.fido_result, path=self.save_path)

        self.time_string = self.parse_time_string(downloaded_files)

        # Delete old images
        # image_folder = join(self.params.local_directory, self.this_name)
        # files = [img for img in listdir(image_folder) if img.endswith(".fits") or img.endswith(".png") ]

        print("Fits File Download Successful from {} to {}".format(self.early, self.now), flush=True)

    def fits_analyze(self):
        """Check several fits files to determine fit curves"""
        print("Analyzing Dataset...", flush=True, end='')
        max_analyze = 5
        minBox = []
        maxBox = []

        self.save_path = join(self.params.local_directory, self.this_name)

        ii = 0
        for filename in listdir(self.save_path):
            if filename.endswith(".fits"):
                ii += 1
                image_path = join(self.save_path, filename)
                fname = filename[3:18]
                fname = fname.replace("_", "")
                time_string = self.parse_time_string(fname, 2)

                # Load the File
                originalData, single_image_data = self.load_fits_series((self.this_name, image_path, time_string))

                self.radial_analyze(originalData, False)
                minBox.append(self.fakeMin)
                maxBox.append(self.fakeMax)

                print('.', end='')
                if ii >= max_analyze:
                    self.fakeMin = np.mean(np.asarray(minBox), axis=0)
                    self.fakeMax = np.mean(np.asarray(maxBox), axis=0)
                    self.fits_analysis_done = True
                    print("Done!")
                    break



    def fits_to_pngs(self):
        """Modify all the Fits images into nice pngs"""

        self.save_path = join(self.params.local_directory, self.this_name)

        files = listdir(self.save_path)
        nFits = len([f for f in files if (f.endswith(".fits") and int(self.time_from_filename(f)[0]) > self.earlyLong)])
        nPng = len([f for f in files if (f.endswith(".png") and int(self.time_from_filename(f)[0]) > self.earlyLong)])
        self.nRem = nFits - nPng
        if self.overwrite_pngs:
            self.nRem = nFits

        if self.nRem > 0:
            print("Generating {} Images...".format(self.nRem), flush=True)
        else:
            print("Images Already Generated")


        ii = 0
        for filename in files:
            if filename.endswith(".fits"):
                pngPath = join(self.save_path, filename[:-4]+'png')
                if not exists(pngPath) or self.overwrite_pngs:
                    # Make the new input
                    time_code, time_string = self.time_from_filename(filename)
                    image_path = join(self.save_path, filename)

                    if int(time_code) < self.earlyLong:
                        if self.remove_old_images:
                            self.deleteFiles(filename)
                        continue

                    ii += 1

                    # Load the File
                    try:
                        originalData, single_image_data = self.load_fits_series((self.this_name, image_path, time_string))
                    except TypeError:
                        remove(image_path)
                        continue


                    # Modify the data
                    data = self.image_modify(originalData)

                    # Plot and save the Data
                    self.plot_and_save(data, single_image_data, originalData, ii)
                continue
            else:
                continue
        if self.nRem > 0:
            print("Done!")

    def deleteFiles(self, filename):
        fitsPath = join(self.save_path, filename)
        pngPath = join(self.save_path, filename[:-4] + 'png')
        remove(fitsPath)
        remove(pngPath)

    def time_from_filename(self, filename):
        fname = filename[3:18]
        time_code = fname.replace("_", "")
        time_string = self.parse_time_string(time_code, 2)
        return time_code, time_string

    def pngs_to_movie(self):
        print("Generating Movie...", end='', flush=True)
        import cv2

        wave = self.this_name
        image_folder = join(self.params.local_directory, wave)
        video_name_mp4 = join(image_folder, '{}_movie.mp4'.format(wave))
        video_name_avi = join(image_folder, '{}_movie.avi'.format(wave))

        try:
            images = [img for img in listdir(image_folder) if (img.endswith(".png") and int(self.time_from_filename(img)[0]) > self.earlyLong)]

            frame = cv2.imread(join(image_folder, images[0]))
            height, width, layers = frame.shape

            fourcc_code = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')

            video_mp4 = cv2.VideoWriter(video_name_mp4, fourcc_code, self.frame_rate, (width, height))
            if self.make_uncompressed:
                video_avi = cv2.VideoWriter(video_name_avi, 0, self.frame_rate, (width, height))

            for image in images:
                im = cv2.imread(join(image_folder, image))
                video_mp4.write(im)
                if self.make_uncompressed:
                    video_avi.write(im)

            cv2.destroyAllWindows()

            video_mp4.release()
            if self.make_uncompressed:
                video_avi.release()

            print('Success!')
        except FileNotFoundError:
            print("Images Not Found")

    def load_fits_series(self, image_data):
        # Load the Fits File from disk

        for ind in np.arange(4):
            try:
                # Parse Inputs
                full_name, save_path, time_string = image_data
                my_map = sunpy.map.Map(save_path)
                break
            except TypeError as e:
                if ind < 3:
                    return np.zeros(1024,1024)
                else:
                    raise e
        data = my_map.data
        return data, image_data

    def image_modify(self, data):
        data = data + 0
        self.radial_analyze(data, False)
        data = self.absqrt(data)
        data = self.coronagraph(data)
        data = self.vignette(data)

        return data

    def plot_and_save(self, data, image_data, original_data=None, ii=None):
        full_name, save_path, time_string = image_data
        name = self.clean_name_string(full_name)

        for processed in [True]:

            if not self.params.is_debug():
                if not processed:
                    continue
            if not processed:
                if original_data is None:
                    continue

            # Create the Figure
            fig, ax = plt.subplots()
            self.blankAxis(ax)

            inches = 10
            fig.set_size_inches((inches, inches))

            pixels = data.shape[0]
            dpi = pixels / inches

            if 'hmi' in name.casefold():
                inst = ""
                plt.imshow(data, origin='upper', interpolation=None)
                # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                plt.tight_layout(pad=5.5)
                height = 1.05

            else:
                inst = '  AIA'
                plt.imshow(data if processed else self.normalize(original_data), cmap='sdoaia{}'.format(name),
                           origin='lower', interpolation=None,
                           vmin=self.vmin_plot, vmax=self.vmax_plot)
                plt.tight_layout(pad=0)
                height = 0.95

            # Annotate with Text
            buffer = '' if len(name) == 3 else '  '
            buffer2 = '    ' if len(name) == 2 else ''

            title = "{}    {} {}, {}{}".format(buffer2, inst, name, time_string, buffer)
            title2 = "{} {}, {}".format(inst, name, time_string)
            ax.annotate(title, (0.125, height + 0.02), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')
            # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
            the_time = strftime("%I:%M%p").lower()
            if the_time[0] == '0':
                the_time = the_time[1:]
            ax.annotate(the_time, (0.125, height), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')

            # Format the Plot and Save
            self.blankAxis(ax)
            middle = '' if processed else "_orig"
            new_path = save_path[:-5] + middle + ".png"


            if ii is not None:
                remString = "{} / {} , {:0.1f}%".format(ii, self.nRem, 100*ii/self.nRem)
            else:
                remString = ""

            try:
                plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
                print("\tSaved {} Image {}, {}".format('Processed' if processed else "Unprocessed", time_string, remString))
            except PermissionError:
                new_path = save_path[:-5] + "_b.png"
                plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
                print("Success")
            except Exception as e:
                print("Failed...using Cached")
                if self.params.is_debug():
                    raise e
            plt.close(fig)

        return new_path

    def update_background(self, local_path, test=False):
        """
        Update the System Background

        Parameters
        ----------
        local_path : str
            The local save location of the image
        """
        print("Updating Background...", end='', flush=True)
        assert isinstance(local_path, str)
        local_path = normpath(local_path)

        this_system = platform.system()

        try:
            if this_system == "Windows":
                import ctypes
                ctypes.windll.user32.SystemParametersInfoW(20, 0, local_path, 0)
                # ctypes.windll.user32.SystemParametersInfoW(19, 0, 'Fill', 0)

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

        if self.params.is_debug():
            self.plot_stats()

        return 0

    # Level 4

    @staticmethod
    def list_files1(directory, extension):
        from os import listdir
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    def get_paths(self, this_result):
        self.name = this_result.get_response(0)[0].wave.wavemin
        while len(self.name) < 4:
            self.name = '0' + self.name
        file_name = '{}_Now.fits'.format(self.name)
        save_path = join(self.params.local_directory, file_name)
        return self.name, save_path

    def import_fido_information(self):
        self.fido_num = self.fido_result.file_num


        self.startTime = self.parse_time_string(str(int(self.fido_result.get_response(0)[0].time.start)), 2)
        self.endTime = self.parse_time_string(str(int(self.fido_result.get_response(0)[-1].time.start)), 2)
        print("\tSearch Found {} images from {} to {}\n".format(self.fido_num, self.startTime, self.endTime), flush=True)

        self.name = self.fido_result.get_response(0)[0].wave.wavemin
        while len(self.name) < 4:
            self.name = '0' + self.name
        save_path = join(self.params.local_directory, self.name)
        makedirs(save_path, exist_ok=True)
        self.save_path = save_path

    @staticmethod
    def parse_time_string(downloaded_files, which=0):
        if which == 0:
            time_string = downloaded_files[0][-25:-10]
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = int(time_string[9:11])
            minute = time_string[11:13]
        else:
            time_string = downloaded_files
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = time_string[8:10]
            minute = time_string[10:12]


        struct_time = (int(year), int(month), int(day), int(hour_raw), int(minute), 0, 0, 0, -1)

        new_time_string = strftime("%I:%M%p %m/%d/%Y ", localtime(timegm(struct_time))).lower()
        if new_time_string[0] == '0':
            new_time_string = new_time_string[1:]

        # print(year, month, day, hour, minute)
        # new_time_string = "{}:{}{} {}/{}/{} ".format(hour, minute, suffix, month, day, year)

        return new_time_string

    @staticmethod
    def clean_name_string(full_name):
        # Make the name strings
        name = full_name + ''
        while name[0] == '0':
            name = name[1:]
        return name

    @staticmethod
    def blankAxis(ax):
        ax.patch.set_alpha(0)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='none', which='both',
                       top=False, bottom=False, left=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Data Manipulations

    @staticmethod
    def reject_outliers(data):
        # # Reject Outliers
        # a = data.flatten()
        # remove_num = 20
        # ind = argpartition(a, -remove_num)[-remove_num:]
        # a[ind] = nanmean(a)*4
        # data = a.reshape(data.shape)

        data[data > 10] = np.nan

        return data

    @staticmethod
    def absqrt(data):
        return np.sqrt(np.abs(data))

    @staticmethod
    def normalize(data):
        high = 98
        low = 15

        lowP = np.nanpercentile(data, low)
        highP = np.nanpercentile(data, high)
        return (data - lowP) / (highP - lowP)

    def vignette(self, data):

        mask = self.radius > (self.noise_radii)
        data[mask] = np.nan
        return data

    def vignette2(self, data):

        mask = np.isclose(self.radius, self.tRadius, atol=2)
        data[mask] = 1

        mask = np.isclose(self.radius, self.noise_radii, atol=2)
        data[mask] = 1
        return data

    def coronagraph(self, data):
        original = sys.stderr
        sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')

        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        dat_corona = (self.dat_flat - self.fakeMin[radius_bin]) / \
                     (self.fakeMax[radius_bin] - self.fakeMin[radius_bin])

        sys.stderr = original

        # Deal with too hot things
        self.vmax = 1
        self.vmax_plot = 1.8

        hot = dat_corona > self.vmax
        dat_corona[hot] = dat_corona[hot] ** (1 / 2)

        # Deal with too cold things
        self.vmin = 0.06
        self.vmin_plot = -0.03

        cold = dat_corona < self.vmin
        dat_corona[cold] = -((np.abs(dat_corona[cold] - self.vmin) + 1) ** (1 / 2) - 1) + self.vmin

        self.dat_coronagraph = dat_corona
        dat_corona_square = dat_corona.reshape(data.shape)

        if self.renew_mask or self.params.mode() == 'r':
            self.corona_mask = self.get_mask(data)
            self.renew_mask = False

        data = self.normalize(data)

        data[self.corona_mask] = dat_corona_square[self.corona_mask]

        # inds = np.argsort(self.rad_flat)
        # rad_sort = self.rad_flat[inds]
        # dat_sort = dat_corona[inds]
        #
        # plt.figure()
        # # plt.yscale('log')
        # plt.scatter(rad_sort[::30], dat_sort[::30], c='k')
        # plt.show()

        return data

    def get_mask(self, dat_out):

        corona_mask = np.full_like(dat_out, False, dtype=bool)
        rezz = corona_mask.shape[0]
        half = int(rezz / 2)

        mode = self.params.mode()

        if type(mode) in [float, int]:
            mask_num = mode
        elif 'y' in mode:
            mask_num = 1
        elif 'n' in mode:
            mask_num = 2
        else:
            if 'r' in mode:
                if len(mode) < 2:
                    mode += 'a'

            if 'a' in mode:
                top = 8
                btm = 1
            elif 'h' in mode:
                top = 6
                btm = 3
            elif 'd' in mode:
                top = 8
                btm = 7
            elif 'w' in mode:
                top = 2
                btm = 1
            else:
                print('Unrecognized Mode')
                top = 8
                btm = 1

            ii = 0
            while True:
                mask_num = np.random.randint(btm, top + 1)
                if mask_num not in self.mask_num:
                    self.mask_num.append(mask_num)
                    break
                ii += 1
                if ii > 10:
                    self.mask_num = []

        if mask_num == 1:
            corona_mask[:, :] = True

        if mask_num == 2:
            corona_mask[:, :] = False

        if mask_num == 3:
            corona_mask[half:, :] = True

        if mask_num == 4:
            corona_mask[:half, :] = True

        if mask_num == 5:
            corona_mask[:, half:] = True

        if mask_num == 6:
            corona_mask[:, :half] = True

        if mask_num == 7:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True

        if mask_num == 8:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True
            corona_mask = np.invert(corona_mask)

        return corona_mask

    # Basic Analysis

    def radial_analyze(self, data, plotStats=False):
        self.offset = np.abs(np.min(data))
        data += self.offset

        self.make_radius(data)
        self.sort_radially(data)
        # self.bin_radially()

        # start1 = time()
        self.bin_stats()
        # print("Old way {:0.3} seconds".format(time()-start1))

        # start2 = time()
        # self.better_bin_stats()
        # print("New way {:0.3} seconds".format(time()-start2))

        # sys.exit()

        if not self.fits_analysis_done:
            self.fit_curves()

        if plotStats:
            self.plot_stats()

    def make_radius(self, data):

        self.rez = data.shape[0]
        centerPt = self.rez / 2
        xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
        xc, yc = xx - centerPt, yy - centerPt

        self.extra_rez = 4

        self.sRadius = 400 * self.extra_rez
        self.tRadius = self.sRadius * 1.28
        self.radius = np.sqrt(xc * xc + yc * yc) * self.extra_rez
        self.rez *= self.extra_rez

    def sort_radially(self, data):
        # Create arrays sorted by radius
        self.rad_flat = self.radius.flatten()
        self.dat_flat = data.flatten()
        inds = np.argsort(self.rad_flat)
        self.rad_sort = self.rad_flat[inds]
        self.dat_sort = self.dat_flat[inds]

    def better_bin_stats(self):

        # Bin the intensities by radius
        # self.radBins = [[] for x in np.arange(self.rez)]
        self.radBins = np.empty((self.rez, self.rez))
        self.radBins.fill(np.nan)
        radCounts = np.zeros(self.rez, dtype=np.int)


        binInds = np.asarray(np.floor(self.rad_sort), dtype=np.int32)


        for ii, binI in enumerate(binInds):
            self.radBins[binI, radCounts[binI]] = self.dat_sort[ii]
            radCounts[binI] += 1
        #
        # length = max(map(len, self.radBins))
        # self.radArray = np.array([xi+[np.NAN]*(length-len(xi)) for xi in self.radBins])


        self.binMin = np.nanpercentile(self.radBins, 2, axis=1)
        self.binMax = np.nanpercentile(self.radBins, 97, axis=1)
        self.binMean = np.nanpercentile(self.radBins, 50, axis=1)


    def bin_stats(self):

        # Bin the intensities by radius
        self.radBins = [[] for x in np.arange(self.rez)]
        binInds = np.asarray(np.floor(self.rad_sort), dtype=np.int32)
        for ii, binI in enumerate(binInds):
            self.radBins[binI].append(self.dat_sort[ii])

        # Find the statistics by bin
        self.binMax = np.zeros(self.rez)
        self.binMin = np.zeros(self.rez)
        self.binMean = np.zeros(self.rez)
        # self.binMed = np.zeros(self.rez)

        for ii, it in enumerate(self.radBins):
            #For each radial bin
            item = np.asarray(it)

            idx = np.isfinite(item)
            finite = item[idx]
            idx2 = np.nonzero(finite - self.offset)
            subItems = finite[idx2]

            if len(subItems) > 0:
                self.binMax[ii] = np.percentile(subItems, 97)  # np.nanmax(subItems)
                self.binMin[ii] = np.percentile(subItems, 2)  # np.min(subItems)
                self.binMean[ii] = np.mean(subItems)
                # self.binMed[ii] = np.median(subItems)
            else:
                self.binMax[ii] = np.nan
                self.binMin[ii] = np.nan
                self.binMean[ii] = np.nan
                # self.binMed[ii] = np.nan

    def fit_curves(self):
        # Input Stuff
        self.radAbss = np.arange(self.rez)
        self.highCut = 730 * self.extra_rez
        theMin = 380 * self.extra_rez
        near_limb = np.arange(theMin, theMin + 50 * self.extra_rez)

        # Find the derivative of the binned mean
        self.diff_mean = np.diff(self.binMean)
        self.diff_mean += np.abs(np.nanmin(self.diff_mean))
        self.diff_mean /= np.nanmean(self.diff_mean) / 100

        # Locate the Limb
        self.limb_radii = np.argmin(self.diff_mean[near_limb]) + theMin
        self.lCut = self.limb_radii - 10 * self.extra_rez
        self.hCut = self.limb_radii + 10 * self.extra_rez

        # Split into three regions
        self.low_abs = self.radAbss[:self.lCut]
        self.low_max = self.binMax[:self.lCut]
        self.low_min = self.binMin[:self.lCut]

        self.mid_abs = self.radAbss[self.lCut:self.hCut]
        self.mid_max = self.binMax[self.lCut:self.hCut]
        self.mid_min = self.binMin[self.lCut:self.hCut]

        self.high_abs = self.radAbss[self.hCut:]
        self.high_max = self.binMax[self.hCut:]
        self.high_min = self.binMin[self.hCut:]

        # Filter the regions separately
        from scipy.signal import savgol_filter

        lWindow = 20 * self.extra_rez + 1
        mWindow = 4 * self.extra_rez + 1
        hWindow = 30 * self.extra_rez + 1
        fWindow = int(3 * self.extra_rez) + 1

        rank = 3

        low_max_filt = savgol_filter(self.low_max, lWindow, rank)

        mid_max_filt = savgol_filter(self.mid_max, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)

        high_max_filt = savgol_filter(self.high_max, hWindow, rank)

        low_min_filt = savgol_filter(self.low_min, lWindow, rank)
        mid_min_filt = savgol_filter(self.mid_min, mWindow, rank)
        high_min_filt = savgol_filter(self.high_min, hWindow, rank)

        # Fit the low curves
        p = np.polyfit(self.low_abs, self.fill_start(low_max_filt), 9)
        low_max_fit = np.polyval(p, self.low_abs)
        p = np.polyfit(self.low_abs, self.fill_start(low_min_filt), 9)
        low_min_fit = np.polyval(p, self.low_abs)

        # Build output curves
        self.fakeAbss = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.fakeMax = np.hstack((low_max_fit, mid_max_filt, high_max_filt))
        self.fakeMin = np.hstack((low_min_fit, mid_min_filt, high_min_filt))

        # Filter again to smooth boundaraies
        self.fakeMax = self.fill_end(self.fill_start(savgol_filter(self.fakeMax, fWindow, rank)))
        self.fakeMin = self.fill_end(self.fill_start(savgol_filter(self.fakeMin, fWindow, rank)))

        # Locate the Noise Floor
        noiseMin = 550 * self.extra_rez - self.hCut
        near_noise = np.arange(noiseMin, noiseMin + 100 * self.extra_rez)
        self.diff_max_abs = self.high_abs[near_noise]
        self.diff_max = np.diff(high_max_filt)[near_noise]
        self.diff_max += np.abs(np.nanmin(self.diff_max))
        self.diff_max /= np.nanmean(self.diff_max) / 100
        self.noise_radii = np.argmin(self.diff_max) + noiseMin + self.hCut
        self.noise_radii = 565 * self.extra_rez

    def fill_end(self, use):
        iii = -1
        val = use[iii]
        while np.isnan(val):
            iii -= 1
            val = use[iii]
        use[iii:] = val
        return use

    def fill_start(self, use):
        iii = 0
        val = use[iii]
        while np.isnan(val):
            iii += 1
            val = use[iii]
        use[:iii] = val
        return use

    def plot_stats(self):

        fig, (ax0, ax1) = plt.subplots(2, 1, True)
        ax0.scatter(self.n2r(self.rad_sort[::30]), self.dat_sort[::30], c='k', s=2)
        ax0.axvline(self.n2r(self.limb_radii), ls='--', label="Limb")
        ax0.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        ax0.axvline(self.n2r(self.lCut), ls=':')
        ax0.axvline(self.n2r(self.hCut), ls=':')
        # ax0.axvline(self.tRadius, c='r')
        ax0.axvline(self.n2r(self.highCut))

        # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
        # plt.plot(self.radAbss[:-1] + 0.5, self.diff_mean, 'r:')

        ax0.plot(self.n2r(self.low_abs), self.low_max, 'm', label="Percentile")
        ax0.plot(self.n2r(self.low_abs), self.low_min, 'm')
        # plt.plot(self.low_abs, self.low_max_fit, 'r')
        # plt.plot(self.low_abs, self.low_min_fit, 'r')

        ax0.plot(self.n2r(self.high_abs), self.high_max, 'c', label="Percentile")
        ax0.plot(self.n2r(self.high_abs), self.high_min, 'c')

        ax0.plot(self.n2r(self.mid_abs), self.mid_max, 'y', label="Percentile")
        ax0.plot(self.n2r(self.mid_abs), self.mid_min, 'y')
        # plt.plot(self.high_abs, self.high_min_fit, 'r')
        # plt.plot(self.high_abs, self.high_max_fit, 'r')

        try:
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMin, 'g')
        except:
            ax0.plot(self.n2r(self.radAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.radAbss), self.fakeMin, 'g')

        # plt.plot(radAbss, binMax, 'c')
        # plt.plot(self.radAbss, self.binMin, 'm')
        # plt.plot(self.radAbss, self.binMean, 'y')
        # plt.plot(radAbss, binMed, 'r')
        # plt.plot(self.radAbss, self.binMax, 'b')
        # plt.plot(radAbss, fakeMin, 'r')
        # plt.ylim((-100, 10**3))
        # plt.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        ax0.set_xlim((0, self.n2r(self.highCut)))
        ax0.legend()
        fig.set_size_inches((8, 12))
        ax0.set_yscale('log')

        ax1.scatter(self.n2r(self.rad_flat[::10]), self.dat_coronagraph[::10], c='k', s=2)
        ax1.set_ylim((-0.25, 2))

        ax1.axhline(self.vmax, c='r', label='Confinement')
        ax1.axhline(self.vmin, c='r')
        ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
        ax1.axhline(self.vmin_plot, c='orange')

        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))

        ax1.legend()
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax0.set_ylabel(r"Absolute Intensity (Counts)")

        plt.tight_layout()
        if self.params.is_debug():
            file_name = '{}_Radial.png'.format(self.name)
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)

            file_name = '{}_Radial_zoom.png'.format(self.name)
            ax0.set_xlim((0.9, 1.1))
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)
            # plt.show()
            plt.close(fig)
        else:
            plt.show()

    def n2r(self, n):
        if True:
            return n / self.limb_radii
        else:
            return n


def run(delay=20, mode='all', debug=False):
    p = Parameters()
    p.mode(mode)
    p.set_delay_seconds(delay)

    if debug:
        p.is_debug(True)
        p.set_delay_seconds(1)
        p.do_HMI(False)

    SunbackMovie(p).start()


def where():
    p = Parameters()
    print(p.discover_best_default_directory())


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    where()
    run(5, 'y', debug=debugg)
