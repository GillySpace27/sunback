from os.path import join
from fetcher.Fetcher import Fetcher
import boto3
import os


class AwsFetcher(Fetcher):
    
    def __init__(self, params):
        self.params = params
        self.params.build_paths_single()
        s3_resource = boto3.resource('s3')
        self.my_bucket = s3_resource.Bucket('gillyspace27-test-billboard')
        self.objects = self.my_bucket.objects.filter(Prefix='renders/')
    
    def fetch(self):
        """Get all the PNGs from the S3 Bucket"""
        print("   Downloading PNGs from S3 to {}".format(self.params.img_directory()))
        for obj in self.objects:
            self.grab(obj)
        print("   All Downloads Complete")
        self.load_imgs()
    
    def grab(self, obj):
        """Get a specific object from the S3 Bucket"""
        
        # Exit if not appropriate find
        if 'orig' in obj.key or 'archive' in obj.key or "thumbs" in obj.key or "4500" in obj.key:
            return
        if self.params.do_one() and self.params.do_one() not in obj.key:
            return
        
        # Identify File
        path, filename = os.path.split(obj.key)
        print('    ', filename)
        loc = join(self.params.img_directory(), filename)
        
        # Download File
        self.my_bucket.download_file(obj.key, loc)
        
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
    #     # find all links on web-page
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
