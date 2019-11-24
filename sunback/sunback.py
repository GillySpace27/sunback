"""
sunback.py
A program that downloads the most current images of the sun from the SDO satellite,
then finds the most likely temperature in each pixel.
Then it sets each of the images to the desktop background in series.

Handles the primary functions
"""

import ctypes
from time import localtime, altzone, timezone, strftime, sleep
from urllib.request import urlretrieve
from os import getcwd, makedirs
from os.path import normpath, abspath, join, dirname
from PIL import Image, ImageFont, ImageDraw
import pytesseract as pt
from tqdm import tqdm

debug = True

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

    @staticmethod
    def download_image(local_path, web_path):
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
        print("Downloading Image...", end='', flush=True)

        try:
            urlretrieve(web_path, local_path)
        except:
            try:
                urlretrieve(web_path, local_path)
            except:
                print("Failed", flush=True)
                raise

        print("Success", flush=True)

    @staticmethod
    def download_file(local_path, web_path):
        print("Downloading Image:", end='\n', flush=True)

        import requests
        # import time
        # now = time.time()
        # print("Get")
        # r = requests.get(web_path, stream=True)
        # print(time.time()-now, flush=True)
        # f = open(local_path, 'wb')
        # file_size = int(r.headers['Content-Length'])
        # for i, chunk in enumerate(r.iter_content()):
        #     f.write(chunk)
        #     print(i//file_size, end="\r")
        # f.close()
        # print(time.time()-now)
        # return
        # import time
        # import json
        # import urllib.request
        # now = time.time()
        # response = urllib.request.urlopen(web_path).read()
        # print("Took {:0.2f} Seconds".format(time.time()-now))

        r = requests.get(web_path, stream=False)
        if r.status_code == 200:
            with open(local_path, 'wb') as f:
                ii=0
                for chunk in r.iter_content(1024):
                    f.write(chunk)
                    print(ii, end='\r', flush=True)
                    ii+=1
        print("\n")

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

        try:
            ctypes.windll.user32.SystemParametersInfoW(20, 0, local_path, 0)
            print("Success")
        except:
            print("Failed")
            raise

    @staticmethod
    def modify_image(local_path, wave, resolution):
        """
        Modify the Image with some Annotations

        Parameters
        ----------
        local_path : str
            The local save location of the image

        wave : str
            The name of the desired wavelength

        resolution: int
            The resolution of the images
        """
        
        print('Modifying Image...', end='', flush=True)

        # Open the image for modification
        img = Image.open(local_path)
        img_raw = img

        try:
            # Are we working with the HMI image?
            is_hmi = wave[0] == 'H'

            # Shrink the HMI images to be the same size
            if is_hmi:
                small_size = int(0.84*resolution)  # 1725
                old_img = img.resize((small_size, small_size))
                old_size = old_img.size

                new_size = (resolution, resolution)
                new_im = Image.new("RGB", new_size)

                x = int((new_size[0] - old_size[0]) / 2)
                y = int((new_size[1] - old_size[1]) / 2)

                new_im.paste(old_img, (x, y))
                img = new_im

            # Read the time and reprint it
            if localtime().tm_isdst:
                offset = altzone / 3600
            else:
                offset = timezone / 3600

            cropped = img_raw.crop((0, 1950, 1024, 2048))
            results = pt.image_to_string(cropped)

            if is_hmi:  # HMI Data
                image_time = results[-6:]
                image_hour = int(image_time[:2])
                image_minute = int(image_time[2:4])

            else:  # AIA Data
                image_time = results[-11:-6]
                image_hour = int(image_time[:2])
                image_minute = int(image_time[-2:])

            image_hour = int(image_hour - offset) % 12
            pre = ''
        except:
            image_hour = localtime().tm_hour % 12
            image_minute = localtime().tm_min
            pre = 'x'

        if image_hour == 0:
            image_hour = 12
        # Draw on the image and save
        draw = ImageDraw.Draw(img)

        # Draw the wavelength
        font = ImageFont.truetype(normpath(r"C:\Windows\Fonts\Arial.ttf"), 42)
        towrite = wave[1:] if wave[0] == '0' else wave
        draw.text((1510, 300), towrite, (200, 200, 200), font=font)

        # Draw a scale Earth
        corner_x = 1580
        corner_y = 350
        width_x = 15
        width_y = width_x
        draw.ellipse((corner_x, corner_y, corner_x + width_x, corner_y + width_y), fill='white', outline='green')

        # Draw the Current Time
        draw.rectangle([(450, 150), (560, 200)], fill=(0, 0, 0))
        draw.text((450, 150), strftime("%I:%M"), (200, 200, 200), font=font)

        # Draw the Image Time
        draw.text((450, 300), "{:0>2}:{:0>2}{}".format(image_hour, image_minute, pre), (200, 200, 200), font=font)

        img.save(local_path)
        print("Success")
        # except:
        #     print("Failed");
        #     return 1
        return 0

    def loop(self):
        """The Main Loop"""
        for wave, web_path in zip(self.params.use_wavelengths, self.params.web_paths):
            print("Image: {}".format(wave))

            # Define the Image
            local_path = self.params.get_local_path(wave)

            # Download the Image
            self.download_file(local_path, web_path)

            # Modify the Image
            self.modify_image(local_path, wave, self.params.resolution)

            # Update the Background
            self.update_background(local_path)

            # Wait for a bit
            self.params.sleep_for_time(wave)


    def run(self):
        """Run the program in a way that won't break"""

        print("\nLive SDO Background Updater \nWritten by Gilly\n")
        print("Resolution: {}\n".format(self.params.resolution))

        while True:
            try:
                self.loop()
            except (KeyboardInterrupt, SystemExit):
                print("Fine, I'll Stop.\n")
                raise
            except:
                print("I failed, but I'm ignoring it.")
                continue

    def debug(self):
        """Run the program in a way that will break"""

        print("\nLive SDO Background Updater \nWritten by Gilly\n")
        print("Resolution: {}\n".format(self.params.resolution))

        while True:
                self.loop()


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
        subdirectory_name = r"data\images"
        if __file__ in globals():
            directory = join(dirname(abspath(__file__)), subdirectory_name)
        else:
            directory = join(abspath(getcwd()), subdirectory_name)
        return directory

    def sleep_for_time(self, wave):
        change = self.background_update_delay_seconds + 0

        if 'temp' in wave:
            change *= self.time_multiplier_for_long_display

        print("Waiting for {} seconds...".format(change), end='', flush=True)

        fps = 10
        for ii in tqdm(range(int(fps*change))):
            sleep(1/fps)
        print("Done\n")


def run(debugg=False):
    p = Parameters()
    p.set_download_resolution(2048)
    if debugg:
        Sunback(p).debug()
    else:
        Sunback(p).run()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run(debug)


