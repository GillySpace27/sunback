import requests
from datetime import datetime
from bs4 import BeautifulSoup
from astropy.io import fits
from tqdm import tqdm
from sunback_filt import Modify
from time import sleep, time
from PIL import Image
import boto3
from os.path import abspath
from platform import system


#Select Amazon Resources
s3 = boto3.resource('s3')
bucket = s3.Bucket('gillyspace27-test-billboard')
s3_client = boto3.client('s3')

#Location of the Solar Images
archive_url = "http://jsoc1.stanford.edu/data/aia/synoptic/nrt/"

#Initialization
global last_time
global start_time
last_time  = time()
start_time = time()
background_update_delay_seconds = 60 #* 60


def lambda_handler_movie(event=None, context=None):
    """is called by aws"""

    print_banner()

    while True:
        try:
            links = modify_img_series()
            sleep_until_delay_elapsed(links)
            # start_time = time()
        except (KeyboardInterrupt, SystemExit):
            print("\n\nOk, I'll Stop. Doot!\n")
            break
        except Exception as e:
            print("Exception: {}".format(e))
            # raise e
        finally:
            pass
            # print('Series Updated at {}'.format(str(datetime.now())))


def print_banner():
    """Prints a message at code start"""
    print("\nSunback Web: SDO Website and Background Updater \nWritten by Chris R. Gilly")
    print("Check out my website: http://gilly.space\n")
    print("Delay: {} Seconds".format(background_update_delay_seconds))


def modify_img_series():
    """Processes the img series"""
    global last_time
    last_time = time()
    print("\nProcessing Images at {}".format(str(datetime.now())[:-7]), flush=True)
    save_times()
    links = []
    for link in tqdm(get_img_links()):
        with fits.open(link, cache=False) as hdul:
            links.append(modify_img(hdul))
    return links


def save_times():
    """Saves the Time file to S3 so we know when images were taken"""
    image_times = requests.get(archive_url+"image_times").text[9:25]
    path = "../aws/image_times"
    with open(path, 'w') as fp:
        fp.write(image_times)
    bucket.upload_file(path, path,
                       ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})


def get_img_links():
    """gets the list of files to pull"""
    # create response object
    r = requests.get(archive_url)

    # create beautiful-soup object
    soup = BeautifulSoup(r.content, 'html5lib')

    # find all links on web-page
    links = soup.findAll('a')

    # filter the link sending with .fits
    img_links = [archive_url + link['href'] for link in links if link['href'].endswith('fits')]
    img_links = [lnk for lnk in img_links if '4500' not in lnk]
    # return [img_links[0]]
    return img_links


def modify_img(hdul):
    """modifies and uploads the image"""
    hdul.verify('silentfix+warn')

    wave, t_rec = hdul[0].header['WAVELNTH'],  hdul[0].header['T_OBS']
    data = hdul[0].data

    image_meta = str(wave), str(wave), t_rec, data.shape
    return upload_imgs(Modify(data, image_meta).get_paths())


def upload_imgs(imgs):
    """uploads all imgs in input to the s3 bucket"""
    for rtPath in imgs:
        smallPath, bigPath, arcPath = make_thumb(rtPath)

        #Upload large File
        bucket.upload_file(rtPath, bigPath,
                           ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})

        #Upload Thumbnail
        bucket.upload_file(smallPath, smallPath,
                           ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})

        #Upload Archive
        if not "orig" in rtPath:
            bucket.upload_file(rtPath, arcPath,
                               ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})

    return smallPath, bigPath, arcPath


def make_thumb(rtPath):
    name = rtPath.split('/')[-1]
    arcPath = "renders/archive/" + "{}_{}".format(int(time()), name)
    smallPath = "renders/thumbs/" + name
    bigPath = 'renders/' + name

    imgDat = Image.open(rtPath)
    imgDat.thumbnail((512,512))
    imgDat.save(smallPath)
    return smallPath, bigPath, arcPath


def sleep_until_delay_elapsed(links):
    """ Make sure that the loop takes the right amount of time """
    wait_if_required(determine_delay(), links)


def determine_delay():
    """ Determine how long to wait """
    delay = background_update_delay_seconds + 0
    global last_time
    # return delay
    run_time_offset = time() - last_time
    delay -= run_time_offset
    delay = max(delay, 0)
    return delay


def wait_if_required(delay, links):
    """ Wait if Required """

    # print("Waiting for {:0.0f} seconds ({} total)".format(delay, background_update_delay_seconds),
    #       flush=True, end='')
    # print('', end='', flush=True)
    # sys.stdout.flush()
    global picNum
    picNum = 0
    for ii in tqdm((range(int(delay))), desc="Waiting for {:0.0f} seconds".format(delay)):
        sleep(1)
        background_handler(ii, links)


def background_handler(ii, links):
    global set_local_background
    global picNum
    if set_local_background:
        if not ii % 60:
            # print(abspath(links[picNum][1]))
            update_background(links[picNum][1])
            picNum += 1
            picNum = picNum % len(links)


def update_background(local_path, test=False):
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
    # print("Updating Background...", end='', flush=True)
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
        # print("Success")
    except Exception as e:
        print("Failed")
        raise e
    #
    # if self.params.is_debug():
    #     self.plot_stats()

    return 0




if __name__ == "__main__":
    # Do something if this file is invoked on its own
    set_local_background = True
    lambda_handler_movie()




