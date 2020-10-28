import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from modify import Modify
from astropy.io import fits
from tqdm import tqdm

# SITE = os.environ['site']  # URL of the site to check, stored in the site environment variable
# EXPECTED = os.environ['expected']  # String expected to be on the page, stored in the expected environment variable
SITE = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"
archive_url = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"
save_path = "fits/"

def lambda_handler(event=None, context=None):
    """is called by aws"""
    try:

        img_links = get_img_links()
        # download_img_series(img_links)
        modify_img_series(img_links)

    except:
        print('Check failed!')
        raise
    else:
        print('Check passed!')
    finally:
        print('Check complete at {}'.format(str(datetime.now())))


def get_img_links():
    """gets the list of files to download"""
    # create response object
    r = requests.get(archive_url)

    # create beautiful-soup object
    soup = BeautifulSoup(r.content,'html5lib')

    # find all links on web-page
    links = soup.findAll('a')

    # filter the link sending with .mp4
    img_links = [archive_url + link['href'] for link in links if link['href'].endswith('fits')]

    return img_links

def download_img_series(img_links):
    """downloads the images"""
    print("Downloading Images", flush=True)
    for link in tqdm(img_links):

        '''iterate through all links in img_links 
        and download them one by one'''

        # obtain filename by splitting url and getting
        # last string
        file_name = link.split('/')[-1]

        # print("Downloading file:%s"%file_name)

        # create response object
        r = requests.get(link, stream = True)

        # download started
        os.makedirs(save_path, exist_ok=True)
        with open(save_path+file_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size = 1024*1024):
                if chunk:
                    f.write(chunk)

        # print("%s downloaded!\n"%file_name)

    print("All imgs downloaded!")
    return

def modify_img_series(img_links):

    for link in img_links:
        file_name = link.split('/')[-1]
        with fits.open(save_path+file_name) as hdul:
            hdul.verify('fix')

            wave = hdul[0].header['WAVELNTH']
            t_rec = hdul[0].header['T_OBS']
            data = hdul[0].data
            image_meta = file_name, file_name, t_rec, 0

            Modify(data, image_meta)



if __name__ == "__main__":
    # Do something if this file is invoked on its own
    lambda_handler()
