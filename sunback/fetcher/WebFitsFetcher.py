import os
import shutil
import sys
import urllib
from datetime import datetime
from os import rename, remove
from os.path import exists, join
from time import time

import numpy as np
import requests
from bs4 import BeautifulSoup
from sunback.fetcher.Fetcher import Fetcher
from tqdm import tqdm
from functools import partial


class WebFitsFetcher(Fetcher):
    base_url = "http://jsoc1.stanford.edu/data/aia/synoptic/mostrecent/"  # Default Location of the Solar Images
    jpg_url_stem = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_{:04}_{:04}.jpg" + "?x=" + str(round(time()))
    description = "Get Fits Files from {}".format(base_url)
    filt_name = "WebFitsFetcher"
    # out_name = 'QRN'
    # name = filt_name = 'QRN Single Shot Processor'
    progress_verb = 'Downloading'
    finished_verb = "Acquired"

    # show_plots = True

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.destroy = True

    def fetch(self, params=None):
        """Gets the Fits Files from the Archive URL
        :param params:
        """
        self.params = params or self.params
        self.load(self.params, quietly=True, wave=self.params.current_wave('rainbow'))
        if self.params.download_files():
            self.__get_img_time()
            paths = self.fetch_fits_files()
            # jpaths = self.fetch_jpegs()
            return paths
        else:
            print("Skipping download!")
            ()

        return self.params.local_fits_paths()

    def prep_for_jpeg_fetch(self):
        if len(self.params.local_fits_paths()) < 1:
            self.load()

        j_urls = []
        rez = None
        for path in self.params.local_fits_paths():
            if rez is None:
                frame, wave, t_rec, center, int_time, nm = self.load_first_fits_field(fits_path=path)
                rez = self.params.rez = frame.shape[0]
            wavenum = int(''.join(i for i in path if i.isdigit()))
            j_urls.append(self.jpg_url_stem.format(self.params.rez, wavenum))

        self.j_directory = os.path.join(self.params.imgs_top_directory(), "jpeg")
        self.j_paths = j_urls

    def fetch_fits_files(self):
        if self.params.get_fits:
            if self.destroy:
                self.delete_directory_items(self.fits_folder)
            print(" V  Downloading Fits Files from {}...".format(self.base_url), flush=True)

            img_links = self.__get_fits_links(self.base_url)
            pbar_iter = tqdm(img_links, desc=" * Downloading Fits")
            if self.params.do_parallel:
                # Run in Parallel
                results = self.params.multi_pool.imap_unordered(self.grab, img_links)
            else:
                # Run in Serial
                results = [self.grab(path) for path in pbar_iter]
            paths = []
            for res in results:
                self.ii += 1
                paths.append(res)
                self.rename_start_frames(res)

            print("\r ^  Successfully Downloaded {} Files\n".format(len(paths)), flush=True)
            return paths

    def fetch_jpegs(self):
        print(" V  Gathering JPEGS...")
        self.print_once = False
        self.prep_for_jpeg_fetch()
        pbar_iter = tqdm(self.j_paths, desc=" * Downloading JPEGs")
        if self.params.do_parallel:
            # Run in Parallel
            results = self.params.multi_pool.imap_unordered(self.grab_jpeg, self.j_paths)
        else:
            # Run in Serial
            results = [self.grab_jpeg(j_path) for j_path in pbar_iter]
        paths = []
        for res in results:
            paths.append(res)
            pbar_iter.update()
            sys.stderr.flush()
        # print("\r ^  DONE!")
        self.params.got_JPEG = True
        return paths

    def grab_jpeg(self, link):
        # print(os.path.basename(link))
        return self.grab(link, directory=self.j_directory)

    def grab(self, link, directory=None):
        tries = 3
        use_temp = False
        filename = link.split('/')[-1]
        filename = filename.split('?')[0]
        use_directory = directory or self.params.fits_directory()
        local_path = join(use_directory, filename)
        # local_temp_path = join(use_directory, "temp", "download__" + filename)
        for ii in np.arange(tries):
            # Retry download
            try:
                self.download_url(link, local_path)
                break
            # except urllib.error.ContentTooShortError:
            # pass
            except Exception as e:
                print("Failed Download...Retrying {} / {}".format(ii, tries))
                print(str(e))
                print(link)
                print(local_path)
                if ii == tries:
                    raise e
        return local_path

    def delete_directory(self, directory):
        if os.path.isdir(directory):
            shutil.rmtree(directory)

    def delete_directory_items(self, directory=None):
        for root, dirs, files in os.walk(directory):
            for file in files:
                self.force_delete(file, root)

    @staticmethod
    def force_delete(file, root='', do=True):
        if do:
            if not os.path.isdir(file):
                os.remove(os.path.join(root, file))
            else:
                shutil.rmtree(file)

        # paths.append(local_path)

    def download_url(self, link, filename=None):

        # return
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        path=None
        import urllib.request as ureq
        if False:
            path = ureq.urlretrieve(link, filename)[0]
        else:
            import certifi
            import ssl
            # contexter = ssl.create_default_context()
            if os.path.exists(filename):
                os.remove(filename)
            with open(filename, "wb") as fp:
                resp = ureq.urlopen(link, cafile=certifi.where()) #context=contexter)
                fp.write(resp.read())
            path = filename


        return path

        # if "fits" in link:
        #     pass

        import shutil
        import urllib.request
        import tempfile

        # # Create a request object with URL and headers
        # url = link
        # header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ', 'Cache-Control': 'max-age=0'}
        # req = urllib.request.Request(url=url, headers=header)
        #
        # # Create an http response object
        # with urllib.request.urlopen(req) as response:
        #     # Create a file object
        #     with open(local_temp_path, "wb") as f:
        #         # Copy the binary content of the response to the file
        #         shutil.copyfileobj(response, f)

        # import urllib3
        # header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ', 'Cache-Control': 'max-age=0'}
        # http = urllib3.PoolManager()
        # # Create a file object
        # print(link)
        # with http.request("GET", link, headers=header, preload_content=False) as r, open(local_temp_path, "wb") as f:
        #     # Copy the binary content of the response to the file
        #     f.write(r.data)
        #     # shutil.copyfileobj(r, f)
        #     r.release_conn()

        # asdf = 1
        # request.data
        # request.add_header('Cache-Control', 'max-age=0')
        # response = urllib3.urlopen(request).read()

        # import requests
        # r = requests.get(link)
        # a = 1
        # req.Request(link, )
        # os.unlink(local_temp_path)
        # ureq.urlcleanup()
        # ureq.urlretrieve(link, local_temp_path)

    @staticmethod
    def __get_fits_links(url):
        """gets the list of files to pull"""
        # create response object
        r = requests.get(url)

        # create beautiful-soup object
        soup = BeautifulSoup(r.content, 'html5lib')

        # not_wanted all links on web-page
        links = soup.findAll('a')

        # filter the link sending with .fits
        img_links = [url + link['href'] for link in links if link['href'].endswith('fits')]
        img_links = [lnk for lnk in img_links if '4500' not in lnk]
        return img_links

    def __get_img_time(self):
        """Gets the time file"""
        image_time = requests.get(self.base_url + "image_times").text[9:25]
        with open(self.params.time_path(), 'w') as fp:
            fp.write(image_time)
