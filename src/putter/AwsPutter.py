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
S3_UPLOAD_ARGS = {'ACL': 'public-read', "ContentDisposition": "inline"}

from copy import copy
txt_args = copy(S3_UPLOAD_ARGS)
txt_args["ContentType"] = "text/plain"

png_args = copy(S3_UPLOAD_ARGS)
png_args["ContentType"] = "image/png"

s3 = boto3.resource('s3')
# bucket = s3.Bucket('gillyspace27-test-billboard')
bucket = s3.Bucket('the-sun-now')
s3_client = boto3.client('s3')
from time import sleep


class AwsPutter(Putter):
    filt_name = "AWSputter"
    description = "Upload Images to AFWS"
    progress_verb = "Uploading"
    progress_verb = "Uploaded"
    progress_unit = "Images"
    
    def put(self, params=None):
        if params is not None:
            self.__init__(params)
        """uploads all imgs in input to the s3 bucket"""
        print(" V Uploading PNGs to {}...".format(bucket), flush=True)
        # sleep(0.1)
    
        self.__save_times()
        self.__upload_files()
    

    def __upload_files(self):
        to_upload = self.params.local_imgs_paths()

        if self.params.do_orig:
            for file in os.listdir(self.params.orig_directory):
                to_upload.append(os.path.join(self.params.orig_directory, file))
                
                
        print("\r * Uploading Files...")
        pbar = tqdm(to_upload, desc=" * Uploading Files")
        for rtPath in pbar:
            pbar.set_description("   + " + os.path.basename(rtPath))
            smallPath, bigPath, arcPath = make_thumbs(rtPath)
            # Upload large File
            bucket.upload_file(rtPath, bigPath,      ExtraArgs=png_args)
    
            # Upload Thumbnail
            bucket.upload_file(smallPath, smallPath, ExtraArgs=png_args)
    
            # Upload Archive
            if "orig" not in rtPath and self.params.do_archive:
                bucket.upload_file(rtPath, arcPath, ExtraArgs=png_args)
        print("\r ^ Success! Uploaded {} PNGs\n".format(len(self.params.local_imgs_paths())))
        
    def __save_times(self):
        """Saves the Time file to S3 so we know when images were taken"""
        print(" * Uploading Time File...", end='')
        path = self.params.time_path()
        path2 = path.replace(".txt", "_readable.txt")
        
        # Read in the Input
        frame, wave, t_rec, center, int_time, nm = self.load_this_fits_frame(self.params.local_fits_paths()[0], self.params.master_frame_list_newest)
        
        # Write the raw output
        # shortened = t_rec.split('.')[0]
        with open(path, "w") as fp:
            fp.write(t_rec)

        tz_list = []
        nzt = self.clean_time_string(t_rec, "NZ"    ).replace("NZDT, ", "NZDT,")
        tz_list.append(nzt)
        tz_list.append(self.clean_time_string(t_rec, 'Japan'   ))
        # tz_list.append(self.clean_time_string(t_rec, "Iran"    ))
        tz_list.append(self.clean_time_string(t_rec, "EET"    ))
        tz_list.append("       ~*~")
        
        # tz_list.append(self.clean_time_string(t_rec, "Europe/Berlin"    ))
        tz_list.append(self.clean_time_string(t_rec, None           ))
        tz_list.append("       ~*~")

        tz_list.append(self.clean_time_string(t_rec, "US/Eastern"   ))
        tz_list.append(self.clean_time_string(t_rec, "US/Mountain"  ))
        tz_list.append(self.clean_time_string(t_rec, "US/Pacific"   ))
        tz_list.append(self.clean_time_string(t_rec, "US/Hawaii"    ))

        # tz_list.append(self.clean_time_string(t_rec, "US/Hawaii"    ))
        

        
        with open(path2, "w") as fp:
            for item in tz_list:
                fp.write(item)
                fp.write("\n")
            
        bucket.upload_file(path, path,   ExtraArgs=txt_args)
        bucket.upload_file(path2, path2, ExtraArgs=txt_args)

        print("Done! ")

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