"""
sunback.py
A program that downloads the most current images of the sun from the SDO satellite,
then finds the most likely temperature in each pixel.
Then it sets each of the images to the desktop background in series.

Handles the primary functions
"""

from time import localtime, altzone, timezone, strftime, sleep, time, asctime
from os import getcwd, makedirs, rename, remove
from os.path import normpath, abspath, join, dirname, exists
from calendar import timegm
import sys
from numpy import sqrt, argpartition, nanmean, abs
import matplotlib.pyplot as plt

from sunpy.net import Fido, attrs as a
import sunpy.cm
import sunpy.map

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

        self.start_time = time()
        self.is_first_run = True

        self.set_default_values()

    def check_real_number(self, number):
        assert type(number) in [float, int]
        assert number > 0

    def set_default_values(self):
        """Sets the Defaults for all the Parameters"""

        #  Set Delay Time for Background Rotation
        self.set_update_delay_seconds(30 * self.seconds)
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
        self.set_file_ending("{}_Now.jpg")

        return 0

    # Methods that Set Parameters
    def set_update_delay_seconds(self, delay):
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
        self.resolution = min([170, 256, 512, 1024, 2048, 3072, 4096], key=lambda x: abs(x - resolution))
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

        subdirectory_name = join("data", "images")
        if __file__ in globals():
            directory = join(dirname(abspath(__file__)), subdirectory_name)
        else:
            directory = join(abspath(getcwd()), subdirectory_name)
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
            print("Waiting for {:0.0f} seconds ({} total)... ".format(delay, self.background_update_delay_seconds),
                  end='', flush=True)

            fps = 10
            for ii in (range(int(fps * delay))):
                sleep(1 / fps)
                # self.check_for_skip()

            print("Done")

    def sleep_until_delay_elapsed(self):
        """ Make sure that the loop takes the right amount of time """
        self.wait_if_required(self.determine_delay())

    # def check_for_skip(self):
    #
    #     import curses
    #     import os
    #
    #     def main(win):
    #         win.nodelay(True)
    #         key = ""
    #         win.clear()
    #         win.addstr("Detected key:")
    #         while 1:
    #             try:
    #                 key = win.getkey()
    #                 win.clear()
    #                 win.addstr("Detected key:")
    #                 win.addstr(str(key))
    #                 if key == os.linesep:
    #                     break
    #             except Exception as e:
    #                 # No input
    #                 pass
    #
    #     curses.wrapper(main)


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
        if parameters:
            self.params = parameters
        else:
            self.params = Parameters()

        self.last_time = 0
        self.this_time = 1
        self.new_images = True
        self.fido_result = None
        self.fido_num = None

    # # Image Stuff

    def fido_search(self):
        """ Find the Most Recent Images """

        minute_range = 18
        self.fido_num = 0

        while not 3 < self.fido_num < 13:
            # Define Time Range
            fmt_str = '%Y/%m/%d %H:%M'
            early = strftime(fmt_str, localtime(time() - minute_range*60 + timezone))
            now = strftime(fmt_str, localtime(time() + timezone))

            # Find Results
            self.fido_result = Fido.search(a.Time(early, now), a.Instrument('aia'))
            self.fido_num = self.fido_result.file_num

            # Change time range if wrong number of records
            if self.fido_num > 13:
                minute_range -= 2
            elif self.fido_num < 3:
                minute_range += 2

            if self.fido_num == 0:
                continue

            self.this_time = int(self.fido_result.get_response(0)[0].time.start)
            self.new_images = self.last_time < self.this_time

        if self.new_images:
            print("Search Found {} new images at {}\n".format(self.fido_num, self.parse_time_string(str(self.this_time),2)), flush=True)
        else:
            print("No New Images, using Cached Data\n")

        self.last_time = self.this_time

        with open(self.params.time_file, 'w') as fp:
            fp.write(str(self.this_time)+'\n')
            fp.write(str(self.fido_num)+'\n')
            fp.write(str(self.fido_result.get_response(0)))

    def fido_retrieve_by_index(self, ind):
        """Retrieve a result by index and save it to file"""

        tries = 3

        if self.new_images:
            for ii in range(tries):
                try:
                    print("Downloading Image...", end='', flush=True)
                    result = self.fido_retrieve_result(self.fido_result[0, ind])
                    print("Success", flush=True)
                    return result
                except KeyboardInterrupt:
                    raise
                except Exception as exp:
                    print("Failed {} Time(s).".format(ii + 1), flush=True)
                    if ii == tries-1:
                        raise exp
        else:
            print("Using Cached Data...", end='', flush=True)
            result = self.list_files1(self.params.local_directory, 'fits')
            print("Success", flush=True)
            file_name = [x for x in result][ind]
            full_name = file_name[:4]

            with open(self.params.time_file, 'r') as fp:
                time_stamp = fp.read()
            time_string = self.parse_time_string(str(time_stamp), 2)
            save_path = join(self.params.local_directory, file_name)
            return full_name, save_path, time_string

    def list_files1(self, directory, extension):
        from os import listdir
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    def fido_get_name_by_index(self, ind):
        name = self.fido_result[0, ind].get_response(0)[0].wave.wavemin
        while len(name)<4:
            name = '0'+ name
        return name

    def fido_retrieve_result(self, this_result):
        """Retrieve a result and save it to file"""
        # Make the File Name
        name = this_result.get_response(0)[0].wave.wavemin
        while len(name)<4:
            name = '0'+ name
        file_name = '{}_Now.fits'.format(name)
        save_path = join(self.params.local_directory, file_name)

        # Download and Rename the File
        original = sys.stderr
        sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w')
        downloaded_files = Fido.fetch(this_result, path=self.params.local_directory)
        sys.stderr = original

        if exists(save_path):
            remove(save_path)

        time_string = self.parse_time_string(downloaded_files)

        rename(downloaded_files[0], save_path)

        return name, save_path, time_string

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

    def fits_to_image(self, image_data):
        """Modify the Fits image into a nice png"""
        print("Modifying Image...", end='', flush=True)
        full_name, save_path, time_string = image_data

        # Make the name strings
        name = full_name + ''
        while name[0] == '0':
            name = name[1:]

        # Create the Figure
        fig, ax = plt.subplots()
        inches = 10
        fig.set_size_inches((inches,inches))

        # Load the Fits File
        my_map = sunpy.map.Map(save_path)

        data = my_map.data
        pixels = my_map.dimensions[0].value
        dpi = pixels / inches

        # Modify the data
        data = sqrt(abs(data))

        # Reject Outliers
        a = data.flatten()
        remove_num = 2
        ind = argpartition(a, -remove_num)[-remove_num:]
        a[ind] = nanmean(a)*4
        data = a.reshape(data.shape)


        # Plot the Data
        plt.imshow(data, cmap='sdoaia{}'.format(name), origin='lower', interpolation=None)

        # Annotate with Text
        buffer = '' if len(name) == 3 else '  '
        buffer2 ='    ' if len(name) == 2 else ''
        title = "{}      AIA {}, {}{}".format(buffer2, name, time_string, buffer)
        title2 = "  AIA {}, {}".format(name, time_string)
        ax.annotate(title, (0.5, 0.95), xycoords='axes fraction', fontsize='large',
                    color='w', horizontalalignment='center')
        ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
        the_time = strftime("%I:%M%p").lower()
        if the_time[0] == '0':
            the_time = the_time[1:]
        ax.annotate(the_time, (0.5, 0.97), xycoords='axes fraction', fontsize='large',
                    color='w', horizontalalignment='center')

        # Format the Plot and Save
        self.blankAxis(ax)
        plt.tight_layout(pad=0)
        new_path = save_path[:-5]+".png"
        plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
        plt.close(fig)

        print("Success")
        return new_path

    def blankAxis(self, ax):
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

    @staticmethod
    def update_background(local_path):
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

        import platform
        this_system = platform.system()

        try:
            if this_system == "Windows":
                import ctypes
                ctypes.windll.user32.SystemParametersInfoW(20, 0, local_path, 0)
                # ctypes.windll.user32.SystemParametersInfoW(19, 0, 'Fill', 0)

            elif this_system == "Darwin":
                from appscript import app, mactypes
                app('Finder').desktop_picture.set(mactypes.File(local_path))
            elif this_system == "Linux":
                import os
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-options 'scaled'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background primary-color 'black'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-uri {}".format(local_path))
            else:
                raise OSError("Operating System Not Supported")
            print("Success")
        except:
            print("Failed")
            raise
        return 0

    # # Main Command Structure

    def run(self):
        """Run the program in a way that won't break"""
        self.print_header(debug=False)

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
                    print("I failed, but I'm ignoring it. Count: {}/{}".format(fail_count, fail_max))
                    continue
                else:
                    print("Too Many Failures, I Quit!")
                    sys.exit(1)

    def debug(self):
        """Run the program in a way that will break"""
        self.print_header(debug=True)

        while True:
            self.execute()

    def print_header(self, debug):
        print("\nSunback: Live SDO Background Updater \nWritten by Chris R. Gilly")
        print("Check out my website: http://gilly.space\n")
        print("Delay: {} Seconds\n".format(self.params.background_update_delay_seconds))
        # print("Resolution: {}\n".format(self.params.resolution))

        if debug:
            print("DEBUG MODE\n")

    def execute(self):
        self.fido_search()
        for ii in range(self.fido_num):
            self.loop(ii)

    def loop(self, ii):
        """The Main Loop"""

        # Gather Data + Print
        self.params.start_time = time()
        this_name = self.fido_get_name_by_index(ii)
        if '94' in this_name and self.params.is_first_run:
            return

        print("Image: {}".format(this_name))

        # Download the Image
        image_data = self.fido_retrieve_by_index(ii)

        # Modify the Image
        image_path = self.fits_to_image(image_data)

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        self.update_background(image_path)

        print('')


def run(delay=20, resolution=2048, debug=False):
    p = Parameters()
    p.set_update_delay_seconds(delay)
    p.set_download_resolution(resolution)

    if debug:
        Sunback(p).debug()
    else:
        Sunback(p).run()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run(20, debug=debugg)











    # def loop_old(self, wave, web_path):
    #     """The Main Loop"""
    #     self.params.start_time = time()
    #     print("Image: {}, at {}".format(wave, asctime()))
    #     # Define the Image
    #     local_path = self.params.get_local_path(wave)
    #
    #     # Download the Image
    #     self.download_image(local_path, web_path)
    #
    #     # Modify the Image
    #     self.modify_image(local_path, wave, self.params.resolution)
    #
    #     # Wait for a bit
    #     self.params.sleep_until_delay_elapsed(wave)
    #
    #     # Update the Background
    #     self.update_background(local_path)
    #
    #     print('')
    #
    #     # print("\n----->>>>>Cycle {:0.1f} seconds\n".format(time()-self.params.start_time))
    # def run_old(self):
    #     """Run the program in a way that won't break"""
    #     self.print_header()
    #
    #     fail_count = 0
    #     fail_max = 10
    #
    #     while True:
    #         for wave, web_path in zip(self.params.use_wavelengths, self.params.web_paths):
    #             try:
    #                 self.loop(wave, web_path)
    #             except (KeyboardInterrupt, SystemExit):
    #                 print("\n\nOk, I'll Stop.\n")
    #                 sys.exit(0)
    #             except Exception as error:
    #                 print("Failure!")
    #                 fail_count += 1
    #                 if fail_count < fail_max:
    #                     print("I failed, but I'm ignoring it. Count: {}/{}".format(fail_count, fail_max))
    #                     continue
    #                 else:
    #                     print("Too Many Failures, I Quit!")
    #                     sys.exit(1)
    # def debug_old(self):
    #     """Run the program in a way that will break"""
    #     self.print_header()
    #
    #     while True:
    #         for wave, web_path in zip(self.params.use_wavelengths, self.params.web_paths):
    #             self.loop(wave, web_path)
    # @staticmethod
    # def modify_image(local_path, wave, resolution):
    #     """
    #     Modify the Image with some Annotations
    #
    #     Parameters
    #     ----------
    #     local_path : str
    #         The local save location of the image
    #
    #     wave : str
    #         The name of the desired wavelength
    #
    #     resolution: int
    #         The resolution of the images
    #     """
    #
    #     print('Modifying Image...', end='', flush=True)
    #
    #     # Open the image for modification
    #     img = Image.open(local_path)
    #     img_raw = img
    #
    #     try:
    #         # Are we working with the HMI image?
    #         is_hmi = wave[0] == 'H'
    #
    #         # Shrink the HMI images to be the same size
    #         if is_hmi:
    #             small_size = int(0.84 * resolution)  # 1725
    #             old_img = img.resize((small_size, small_size))
    #             old_size = old_img.size
    #
    #             new_size = (resolution, resolution)
    #             new_im = Image.new("RGB", new_size)
    #
    #             x = int((new_size[0] - old_size[0]) / 2)
    #             y = int((new_size[1] - old_size[1]) / 2)
    #
    #             new_im.paste(old_img, (x, y))
    #             img = new_im
    #
    #         # Read the time and reprint it
    #         if localtime().tm_isdst:
    #             offset = altzone / 3600
    #         else:
    #             offset = timezone / 3600
    #
    #         cropped = img_raw.crop((0, 1950, 1024, 2048))
    #         results = pt.image_to_string(cropped)
    #
    #         if is_hmi:  # HMI Data
    #             image_time = results[-6:]
    #             image_hour = int(image_time[:2])
    #             image_minute = int(image_time[2:4])
    #
    #         else:  # AIA Data
    #             image_time = results[-11:-6]
    #             image_hour = int(image_time[:2])
    #             image_minute = int(image_time[-2:])
    #
    #         image_hour = int(image_hour - offset) % 12
    #         pre = ''
    #     except:
    #         image_hour = localtime().tm_hour % 12
    #         image_minute = localtime().tm_min
    #         pre = 'x'
    #
    #     if image_hour == 0:
    #         image_hour = 12
    #     # Draw on the image and save
    #     draw = ImageDraw.Draw(img)
    #
    #     # Draw the wavelength
    #     font = ImageFont.truetype(normpath(r"C:\Windows\Fonts\Arial.ttf"), 42)
    #     towrite = wave[1:] if wave[0] == '0' else wave
    #     draw.text((1510, 300), towrite, (200, 200, 200), font=font)
    #
    #     # Draw a scale Earth
    #     corner_x = 1580
    #     corner_y = 350
    #     width_x = 15
    #     width_y = width_x
    #     draw.ellipse((corner_x, corner_y, corner_x + width_x, corner_y + width_y), fill='white', outline='green')
    #
    #     # Draw the Current Time
    #     draw.rectangle([(450, 150), (560, 200)], fill=(0, 0, 0))
    #     draw.text((450, 150), strftime("%I:%M"), (200, 200, 200), font=font)
    #
    #     # Draw the Image Time
    #     draw.text((450, 300), "{:0>2}:{:0>2}{}".format(image_hour, image_minute, pre), (200, 200, 200), font=font)
    #
    #     img.save(local_path)
    #     print("Success")
    #     # except:
    #     #     print("Failed");
    #     #     return 1
    #     return 0
    #
    # def download_image(self, local_path, web_path):
    #     """
    #     Download an image and save it to file
    #
    #     Go to the internet and download an image
    #
    #     Parameters
    #     ----------
    #     web_path : str
    #         The web location of the image
    #
    #     local_path : str
    #         The local save location of the image
    #     """
    #     tries = 3
    #
    #     for ii in range(tries):
    #         try:
    #             print("Downloading Image...", end='', flush=True)
    #             urlretrieve(web_path, local_path)
    #             print("Success", flush=True)
    #             return 0
    #         except KeyboardInterrupt:
    #             raise
    #         except Exception as exp:
    #             print("Failed {} Time(s).".format(ii + 1), flush=True)
    #             if ii == tries-1:
    #                 raise exp

    # def fido_retrieve_all(self):
    #     """Retrieve all searched results and save them to file"""
    #     all_downloads = []
    #     for ii in range(self.fido_num):
    #         all_downloads.append(self.fido_retrieve_by_index(ii))
    #     return all_downloads