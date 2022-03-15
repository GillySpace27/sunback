from os.path import split
from os import makedirs
from time import time

from tqdm import tqdm
from putter.Putter import Putter
import boto3

# Select Amazon Resources
# from utils.file_util import get_thumblinks
from utils.array_util import get_thumblinks, make_thumbs
import os
S3_UPLOAD_ARGS = {'ACL': 'public-read', "ContentType": "image/png", "ContentDisposition": "inline"}
s3 = boto3.resource('s3')
bucket = s3.Bucket('gillyspace27-test-billboard')
s3_client = boto3.client('s3')
from time import sleep


class AwsPutter(Putter):
    
    def put(self, params=None):
        if params is not None:
            self.__init__(params)
        """uploads all imgs in input to the s3 bucket"""
        print("  *Uploading PNGs to {}...".format(bucket), flush=True)
        sleep(0.1)
    
        to_upload = self.params.local_imgs_paths()
    
        if self.params.do_orig:
            for file in os.listdir(self.params.orig_directory):
                to_upload.append(os.path.join(self.params.orig_directory, file))
        pbar = tqdm(to_upload, desc="Uploading Files")
        for rtPath in pbar:
            pbar.set_description(os.path.basename(rtPath))
            smallPath, bigPath, arcPath = make_thumbs(rtPath)

            # Upload large File
            bucket.upload_file(rtPath, bigPath, ExtraArgs=S3_UPLOAD_ARGS)
        
            # Upload Thumbnail
            bucket.upload_file(smallPath, smallPath, ExtraArgs=S3_UPLOAD_ARGS)
        
            # Upload Archive
            if "orig" not in rtPath and self.params.do_archive:
                bucket.upload_file(rtPath, arcPath, ExtraArgs=S3_UPLOAD_ARGS)
        self.__save_times()
        print("  Success! Uploaded {} PNGs\n".format(len(self.params.local_imgs_paths())))
        

        
    def __save_times(self):
        """Saves the Time file to S3 so we know when images were taken"""
        path = self.params.time_path()
        bucket.upload_file(path, path, ExtraArgs=S3_UPLOAD_ARGS)


    # def put_ultimate(self):
    #     """uploads all imgs in input to the s3 bucket"""
    #     print("   Uploading files to {}...".format(bucket), flush=True)
    #     sleep(0.1)
    #     for local, remote in tqdm(self.params.local_imgs_paths()):
    #
    #         # Upload file
    #         bucket.upload_file(local, remote, ExtraArgs=S3_UPLOAD_ARGS)
    #
    #     self.__save_times()
    #     print("  Success! Uploaded {} files\n".format(len(self.params.local_imgs_paths())))