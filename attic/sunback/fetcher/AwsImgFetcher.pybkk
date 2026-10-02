import sys
from os.path import join

from tqdm import tqdm

from src.fetcher.Fetcher import Fetcher
import boto3
import os


class AwsImgFetcher(Fetcher):
    description = "Get imgs from the Amazon S3 Bucket"

    def __init__(self, params=None, quick=False, rp=None):
        # Initialize class variables
        super().__init__(params, quick, rp)
        if self.params:
            # self.load(params)
            s3_resource = boto3.resource('s3')
            self.my_bucket = s3_resource.Bucket('gillyspace27-test-billboard')
            self.objects = self.my_bucket.objects.filter(Prefix='renders/')
            self.n_obj = 0

    def fetch(self, params=None):
        """Get all the PNGs from the S3 Bucket
        :param params:
        """
        self.__init__(params)
        sys.stdout.flush()
        print("   Downloading PNGs from Amazon S3 to {}".format(self.params.imgs_top_directory()), flush=True)
        for ii, obj in enumerate(self.objects):
            self.grab_obj(obj)

        self.load()

        if self.n_imgs >= ii:
            print("\r   All Downloads Complete", flush=True)
        elif len(self.params.imgs_top_directory()) == 0:
            print("\r     No Files Loaded", flush=True)
        else:print("\r     {} Files Loaded".format(self.n_imgs), flush=True)

        sys.stdout.flush()

    def grab_obj(self, obj):
        """Get a specific object from the S3 Bucket"""

        # Exit if not appropriate not_wanted
        if 'orig' in obj.key or 'archive' in obj.key or "thumbs" in obj.key or "4500" in obj.key:
            return
        if self.params.do_one() and self.params.do_one() not in obj.key:
            return

        # Identify File
        path, filename = os.path.split(obj.key)
        # print(filename, pointing_end=', ')
        loc = join(self.params.imgs_top_directory(), "dl_" + filename)

        # Download File
        self.my_bucket.download_file(obj.key, loc)
        print('\r     ', end='')
        print(obj.key, end='', flush=True)
        sys.stdout.flush()
        return



    # @staticmethod
    # def __get_fits_links(url):
    #     """gets the list of files to pull"""
    #     # create response object
    #     r = requests.get(url)
    #
    #     # create beautiful-soup object
    #     soup = BeautifulSoup(r.content, 'html5lib')
    #
    #     # not_wanted all links on web-page
    #     links = soup.findAll('a')
    #
    #     # filter the link sending with .fits
    #     img_links = [archive_url + link['href'] for link in links if link['href'].endswith('fits')]
    #     img_links = [lnk for lnk in img_links if '4500' not in lnk]
    #     return img_links
    #
    # def __get_img_time(self):
    #     """Gets the time file"""
    #     image_time = requests.get(archive_url + "image_times").text[9:25]
    #     with open(self.params.time_path(), 'w') as fp:
    #         fp.write(image_time)
