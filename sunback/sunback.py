"""
sunback.py
A program that downloads the most current images of the sun from the SDO satellite, then finds the most likely temperature in each pixel. Then it sets each of the images to the desktop background in series. 

Handles the primary functions
"""

import ctypes
import time
import pathlib

from urllib.request import urlretrieve as urlret
from random import shuffle
import os.path as path

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import pytesseract as tes
from numpy import mod
from numpy import floor

# import aiaTemp_parallel as aia

# import sys, os
import sys

sys.stderr = open("errlog.txt", "w")


# Function Definitions

def download_image(web_path, local_path, wave):
    """
    Download an image and save it to file asdf

    Go to the internet and download an image

    Parameters
    ----------
    web_path : str
        The web location of the image repository
        
    local_path : str
        The local save location of the image
        
    wave : str
        The name of the desired image

    Returns
    -------
    success : bool
        0 if success
    """

    print("Downloading Image...", end='', flush=True)
    # if wave[0] == 't':
    #     try:
    #         aia.runParallel()
    #     except:
    #         try:
    #             aia.runParallel()
    #         except:
    #             print('Failed', flush=True)
    #             return 1
    # else:
    try:
        urlret(web_path, local_path)
    except:
        try:
            urlret(web_path, local_path)
        except:
            print('Failed', flush=True);
            return 1
    print("Success", flush=True)
    return 0


def update_background(local_path):
    """Update the system background"""
    try:
        ctypes.windll.user32.SystemParametersInfoW(20, 0, local_path, 0)
        print("Background Updated")
    except:
        print("Failed to update background");
        raise


def save_reference(pathToFile, wave):
    """Grab a single reference image and save"""
    try:
        print("Saving Reference Image...", end='', flush=True)
        now = time.strftime("%y-%j-%H-%M")
        webpath = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_{}.jpg".format(wave)
        fullpath = path.normpath(pathToFile + "reference\\171_reference_{}.jpg".format(now))
        urlret(webpath, fullpath)
        print("Success")
    except:
        print("Failed")


def imageMod(fullPath, wave, resolution):
    """Modify the image"""
    try:
        print('Modifying Image...', end='', flush=True)
        # Open the image for modification
        img = Image.open(fullPath)
        img_raw = img
        font = ImageFont.truetype(path.normpath("C:\Windows\Fonts\Arial.ttf"), 42)

        # Shrink the HMI images to be the same size
        if wave[0] == 'H':
            smallSize = int(0.84*resolution) #1725
            print(smallSize)
            old_img = img.resize((smallSize, smallSize))
            old_size = old_img.size

            new_size = (resolution, resolution)
            new_im = Image.new("RGB", new_size)

            x = int((new_size[0] - old_size[0]) / 2)
            y = int((new_size[1] - old_size[1]) / 2)

            new_im.paste(old_img, (x, y))
            img = new_im

        # Read the time and reprint it
        if time.localtime().tm_isdst:
            offset = time.altzone / hours
        else:
            offset = time.timezone / hours

        # try:
        cropped = img_raw.crop((0, 1950, 1024, 2048))

        # cornerX = 218
        # cornerY = 970
        # widX = 1612
        # widY = 100
        # cropped2 = img_raw.crop((cornerX,cornerY, cornerX + widX, cornerY+widY))
        # cropped2.show()

        results = tes.image_to_string(cropped)

        if wave[0] == 'H':  # HMI Data
            imgTime = results[-6:]
            imgHour = int(imgTime[:2])
            imgMin = int(imgTime[2:4])

        else:  # AIA Data
            imgTime = results[-11:-6]
            imgHour = int(imgTime[:2])
            imgMin = int(imgTime[-2:])

        imgHour = int(mod(imgHour - offset, 12))
        pre = ''
    except:
        imgHour = mod(time.localtime().tm_hour, 12)
        imgMin = time.localtime().tm_min
        pre = 'x'

    if imgHour == 0: imgHour = 12
    # Draw on the image and save
    draw = ImageDraw.Draw(img)

    # Draw the wavelength
    towrite = wave[1:] if wave[0] == '0' else wave
    draw.text((1510, 300), towrite, (200, 200, 200), font=font)

    # Draw a scale Earth
    cornerX = 1580
    cornerY = 350
    widX = 15
    widY = widX
    draw.ellipse((cornerX, cornerY, cornerX + widX, cornerY + widY), fill='white', outline='green')

    # Draw the Current Time
    draw.rectangle([(450, 150), (560, 200)], fill=(0, 0, 0))
    draw.text((450, 150), time.strftime("%I:%M"), (200, 200, 200), font=font)

    # Draw the Image Time
    draw.text((450, 300), "{:0>2}:{:0>2}{}".format(imgHour, imgMin, pre), (200, 200, 200), font=font)

    img.save(fullPath)
    print("Success")
    # except:
    #     print("Failed");
    #     return 1
    return 0


def run():
    # ##Set up the variables

    #  Delay Time for Background Rotation
    seconds = 1
    minutes = 60
    hours = minutes * 60
    picChangeTime = 30 * seconds  # 0.3 * minutes
    longMultiple = 3

    # File Paths
    pathToFile = r"C:\Users\chgi7364\Dropbox\Drive\Pictures\SunToday\\"

    # pathToFile = path.dirname(path.abspath(__file__)) + "\Images\\"
    pathlib.Path(pathToFile).mkdir(parents=True, exist_ok=True)

    # Wavelength Selection

    resolution = 2048
    webStringAIA = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_{}{}".format(resolution, "_{}.jpg")
    wavelengths = ['0171', '0193', '0211', '0304', '0131', '0335', '0094', 'HMIBC', 'HMIIF']
    # wavelengths = ['0193', 'temp']
    wavelengths.sort()
    # wavelengths.insert(0, 'temp')

    webPaths = [webStringAIA.format(wave) for wave in wavelengths]
    webPaths.append("https://sdo.gsfc.nasa.gov/assets/img/latest/f_211_193_171pfss_{}.jpg".format(resolution))
    wavelengths.append("PFSS")

    fileEnd = "_Now.jpg"

    ##The Main Loop

    print("\nLive SDO Background Updater \nWritten by Gilly\n")
    print("Resolution: {}".format(resolution))

    while True:

        for wave, webPath in zip(wavelengths, webPaths):
            # try:
            # Define the Image
            print("Image: {}".format(wave))
            fullPath = path.normpath(pathToFile + wave + fileEnd)

            # Download the Image
            if download_image(webPath, fullPath, wave):
                continue

            # Modify the Image
            if imageMod(fullPath, wave, resolution):
                pass

            # Update the Background
            update_background(fullPath)

            # Wait for a bit
            if 'temp' in wave:
                change = longMultiple * picChangeTime
            else:
                change = picChangeTime

            print("Waiting for {} seconds...".format(change), end='', flush=True)
            time.sleep(change)
            print("Done\n")
            # except (KeyboardInterrupt, SystemExit):
            #     print("Fine, I'll Stop.\n");
            #     raise
            # except:
            #     print("I failed")

    # save_reference(pathToFile, '0211')



if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run()


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote
