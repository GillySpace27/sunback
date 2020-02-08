"""
sunback.py
A program that downloads the most current images of the sun from the SDO satellite,
then sets each of the images to the desktop background in series.

Handles the primary functions
"""

# Imports
from time import localtime, timezone, strftime, sleep, time
from urllib.request import urlretrieve
from os import getcwd, makedirs, rename, remove
from os.path import normpath, abspath, join, dirname, exists
from calendar import timegm
from sunpy.net import Fido, attrs as a
import sunpy.map
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

debugg = False


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
        self.mask_num = [2]

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
                print("\n\nOk, I'll Stop.\n")
                sys.exit(0)
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
            self.run_HMI()
            self.indexNow = 0
            self.run_HMI_white()
        self.indexNow = 0

    # # Main Loops

    def run_AIA(self, ii):
        """The Main Loop"""

        # Gather Data + Print
        this_name = self.fido_get_name_by_index(ii)

        if '4500' in this_name:
            return
        if '94' in this_name and self.params.is_first_run:
            # print("Skip for Now\n")
            return

        print("Image: {}".format(this_name))

        # Retrieve and Prepare the Image
        image_path = self.make_image(ii)

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        self.update_background(image_path)

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
        new_path = self.plot_and_save(mpimg.imread(self.hmi_path), ('HMI', self.hmi_path, "Magnetic Field", -1))

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
                break

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

        if self.new_images:
            print("Search Found {} new images at {}\n".format(self.fido_num,
                                                              self.parse_time_string(str(self.this_time), 2)),
                  flush=True)
        else:
            print("No New Images, using Cached Data\n")

        self.last_time = self.this_time

        with open(self.params.time_file, 'w') as fp:
            fp.write(str(self.this_time) + '\n')
            fp.write(str(self.fido_num) + '\n')
            fp.write(str(self.fido_result.get_response(0)))

    def fido_get_name_by_index(self, ind):
        self.params.start_time = time()
        name = self.fido_result[0, ind].get_response(0)[0].wave.wavemin
        while len(name) < 4:
            name = '0' + name
        return name

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
        print("Generating Image...", end='', flush=True)

        data, image_data = self.load_fits(image_data)

        # Modify the data
        data = self.image_modify(data)

        # Plot the Data
        new_path = self.plot_and_save(data, image_data)

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

        self.radial_analyze(data, False)
        data = self.absqrt(data)
        data = self.normalize(data)
        data = self.coronagraph(data)
        data = self.vignette(data)

        return data

    def plot_and_save(self, data, image_data):
        full_name, save_path, time_string, ii = image_data
        name = self.clean_name_string(full_name)

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
            plt.imshow(data, cmap='sdoaia{}'.format(name), origin='lower', interpolation=None, vmin=0, vmax=2)
            plt.tight_layout(pad=0)
            height = 0.95

        # Annotate with Text
        buffer = '' if len(name) == 3 else '  '
        buffer2 ='    ' if len(name) == 2 else ''


        title = "{}    {} {}, {}{}".format(buffer2, inst, name, time_string, buffer)
        title2 = "{} {}, {}".format(inst, name, time_string)
        ax.annotate(title, (0.5, height), xycoords='axes fraction', fontsize='large',
                    color='w', horizontalalignment='center')
        ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
        the_time = strftime("%I:%M%p").lower()
        if the_time[0] == '0':
            the_time = the_time[1:]
        ax.annotate(the_time, (0.5, height + 0.02), xycoords='axes fraction', fontsize='large',
                    color='w', horizontalalignment='center')

        # Format the Plot and Save
        self.blankAxis(ax)
        new_path = save_path[:-5]+".png"

        try:
            plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
            print("Success")
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

    @staticmethod
    def update_background(local_path, test=False):
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
        return 0


    # Level 4

    @staticmethod
    def list_files1(directory, extension):
        from os import listdir
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    def get_paths(self, this_result):
        name = this_result.get_response(0)[0].wave.wavemin
        while len(name)<4:
            name = '0'+ name
        file_name = '{}_Now.fits'.format(name)
        save_path = join(self.params.local_directory, file_name)
        return name, save_path

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
        high = 99.5
        low = 4

        lowP = np.nanpercentile(data, low)
        highP = np.nanpercentile(data, high)
        return (data - lowP) / (highP - lowP)

    def vignette(self, data):

        mask = self.radius > (self.sRadius*1.28)
        data[mask] = np.nan
        # import pdb; pdb.set_trace()
        return data

    def coronagraph(self, data):
        original = sys.stderr
        sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')

        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        dat_corona = (self.dat_flat - self.fakeMin[radius_bin]) / \
                         (self.fakeMax[radius_bin] - self.fakeMin[radius_bin])

        sys.stderr = original

        dat_corona_square = dat_corona.reshape(data.shape)

        if self.renew_mask:
            self.corona_mask = self.get_mask(data)
            self.renew_mask = False

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
        self.make_radius(data)
        self.sort_radially(data)
        self.bin_radially()
        self.fit_curves()

        if plotStats:
            self.plot_stats()

    def make_radius(self, data):
        self.sRadius = 400

        self.rez = data.shape[0]
        centerPt = self.rez / 2
        xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
        xc, yc = xx - centerPt, yy - centerPt
        self.radius = np.sqrt(xc * xc + yc * yc)

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
            subItems = item[idx]
            if len(subItems)>0:
                self.binMax[ii] = np.percentile(subItems, 99) #np.nanmax(subItems)
                self.binMin[ii] = np.percentile(subItems, 0.5) #np.min(subItems)
                self.binMean[ii] = np.mean(subItems)
                self.binMed[ii] = np.median(subItems)
            else:
                self.binMax[ii] = np.nan
                self.binMin[ii] = np.nan
                self.binMean[ii] = np.nan
                self.binMed[ii] = np.nan

    def fit_curves(self):
        self.limb_radii = np.argmax(self.binMean[300:500]) + 300
        self.radAbss = np.arange(self.rez)

        self.highCut = 730

        self.low_abs = self.radAbss[:self.limb_radii]
        self.low_max = self.binMax[:self.limb_radii]
        self.low_min = self.binMin[:self.limb_radii]

        p = np.polyfit(self.low_abs, self.low_max, 3)
        self.low_max_fit = np.polyval(p, self.low_abs)
        p = np.polyfit(self.low_abs, self.low_min, 5)
        self.low_min_fit = np.polyval(p, self.low_abs)

        self.high_abs = self.radAbss[self.limb_radii:self.highCut]
        self.high_max = self.binMax[self.limb_radii:self.highCut]
        self.high_min = self.binMin[self.limb_radii:self.highCut]

        idx = np.isfinite(self.high_abs) & np.isfinite(self.high_max)
        p = np.polyfit(self.high_abs[idx], self.high_max[idx], 2)
        self.high_max_fit = np.polyval(p, self.high_abs)

        idx = np.isfinite(self.high_abs) & np.isfinite(self.high_min)
        p = np.polyfit(self.high_abs[idx], self.high_min[idx], 2)
        self.high_min_fit = np.polyval(p, self.high_abs)

        self.radAbss2 = np.hstack((self.low_abs, self.high_abs))
        self.fakeMax = np.hstack((self.low_max_fit, self.high_max))
        self.fakeMin = np.hstack((self.low_min_fit, self.high_min))

    def plot_stats(self):

        plt.figure()
        # plt.yscale('log')
        plt.scatter(self.rad_sort[::30], self.dat_sort[::30], c='k')
        plt.axvline(self.limb_radii)

        # plt.plot(self.low_abs, self.low_max, 'm')
        # plt.plot(self.low_abs, self.low_min, 'm')
        # plt.plot(self.low_abs, self.low_max_fit, 'r')
        # plt.plot(self.low_abs, self.low_min_fit, 'r')
        #
        # plt.plot(self.high_abs, self.high_max, 'c')
        # plt.plot(self.high_abs, self.high_min, 'c')
        # plt.plot(self.high_abs, self.high_min_fit, 'b')
        # plt.plot(self.high_abs, self.high_max_fit, 'b')

        plt.plot(self.radAbss2, self.fakeMax, 'g')
        plt.plot(self.radAbss2, self.fakeMin, 'g')


        # plt.plot(radAbss, binMax, 'c')
        # plt.plot(self.radAbss, self.binMin, 'm')
        # plt.plot(self.radAbss, self.binMean, 'y')
        # plt.plot(radAbss, binMed, 'r')
        # plt.plot(self.radAbss, self.binMax, 'b')
        # plt.plot(radAbss, fakeMin, 'r')

        plt.show(True)


def run(delay=20, mode='all', debug=False):
    p = Parameters()
    p.mode(mode)
    p.set_delay_seconds(delay)

    if debug:
        p.is_debug(True)
        p.set_delay_seconds(1)
        p.do_HMI(False)

    Sunback(p).start()


def where():
    p = Parameters()
    print(p.discover_best_default_directory())


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run(20, debug=debugg)
