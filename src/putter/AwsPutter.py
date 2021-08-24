from os.path import split
from os import makedirs
from time import time

from tqdm import tqdm
from putter.Putter import Putter
import boto3

# Select Amazon Resources
# from utils.file_util import get_thumblinks
from utils.array_util import get_thumblinks

S3_UPLOAD_ARGS = {'ACL': 'public-read', "ContentType": "image/png"}
s3 = boto3.resource('s3')
bucket = s3.Bucket('gillyspace27-test-billboard')
s3_client = boto3.client('s3')
from time import sleep


class AwsPutter(Putter):
    def __init__(self, params):
        self.params = params
    
    def put(self):
        """uploads all imgs in input to the s3 bucket"""
        print("Uploading PNGs to {}...".format(bucket), flush=True)
        sleep(0.1)
        for rtPath in tqdm(self.params.local_img_paths()):
            smallPath, bigPath, arcPath = get_thumblinks(rtPath)
            
            # Upload large File
            bucket.upload_file(rtPath, bigPath, ExtraArgs=S3_UPLOAD_ARGS)
            
            # Upload Thumbnail
            bucket.upload_file(smallPath, smallPath, ExtraArgs=S3_UPLOAD_ARGS)
            
            # Upload Archive
            if "orig" not in rtPath:
                bucket.upload_file(rtPath, arcPath, ExtraArgs=S3_UPLOAD_ARGS)
        self.__save_times()
        print("Success! Uploaded {} PNGs\n".format(len(self.params.local_img_paths())))
        
    def put_ultimate(self):
        """uploads all imgs in input to the s3 bucket"""
        print("Uploading files to {}...".format(bucket), flush=True)
        sleep(0.1)
        for local, remote in tqdm(self.params.local_img_paths()):
            
            # Upload file
            bucket.upload_file(local, remote, ExtraArgs=S3_UPLOAD_ARGS)

        self.__save_times()
        print("Success! Uploaded {} files\n".format(len(self.params.local_img_paths())))
        
    def __save_times(self):
        """Saves the Time file to S3 so we know when images were taken"""
        path = self.params.time_path()
        bucket.upload_file(path, path, ExtraArgs=S3_UPLOAD_ARGS)
