from os.path import split
from os import makedirs
from time import time

from tqdm import tqdm
from Executor.ModifyExecutor import ModifyExecutor as me
from Putter.Putter import Putter
import boto3
import requests

# Select Amazon Resources
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
            smallPath, bigPath, arcPath = me.get_thumblinks(rtPath)
            
            # Upload large File
            bucket.upload_file(rtPath, bigPath,
                               ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})
            
            # Upload Thumbnail
            bucket.upload_file(smallPath, smallPath,
                               ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})
            
            # Upload Archive
            if "orig" not in rtPath:
                bucket.upload_file(rtPath, arcPath,
                                   ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})
        self.__save_times()
        print("Success! Uploaded {} PNGs\n".format(len(self.params.local_img_paths())))
        
    

    def __get_thumblinks(self, rtPath):
        name = split(rtPath)[-1]
        arcPath = "renders/archive/" + "{}_{}".format(int(time()), name)
        smallPath = "renders/thumbs/" + name
        bigPath = 'renders/' + name
        makedirs("renders/thumbs/", exist_ok=True)
        return smallPath, bigPath, arcPath

    
    def __save_times(self):
        """Saves the Time file to S3 so we know when images were taken"""
        path = self.params.time_path()
        bucket.upload_file(path, path,
                           ExtraArgs={'ACL': 'public-read', "ContentType": "image/png"})
