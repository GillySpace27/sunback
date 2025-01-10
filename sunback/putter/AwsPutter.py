from os.path import split
from os import makedirs
from time import time

from tqdm import tqdm
from sunback.putter.Putter import Putter
import boto3

# Select Amazon Resources
# from src.utils.file_util import get_thumblinks
from sunback.utils.array_util import get_thumblinks, make_thumbs
import os
S3_UPLOAD_ARGS = {'ACL': 'public-read', "ContentDisposition": "inline"}

from copy import copy
txt_args = copy(S3_UPLOAD_ARGS)
txt_args["ContentType"] = "text/plain"

png_args = copy(S3_UPLOAD_ARGS)
png_args["ContentType"] = "image/png"

s3 = boto3.resource('s3')
# bucket = s3.Bucket('gillyspace27-test-billboard')
bucket_name = 'the-sun-now'
bucket = s3.Bucket(bucket_name)
s3_client = boto3.client('s3')
from time import sleep


class AwsPutter(Putter):
    filt_name = "AWSputter"
    description = "Upload Images to AWS {}".format(bucket_name)
    progress_verb = "Uploading"
    progress_verb = "Uploaded"
    progress_unit = "Images"

    def __init__(self, params=None, quick=False, rp=None, in_name=None):
        super().__init__(params, quick, rp, in_name)

        self.pbar = None
        self.to_upload = None

    def put(self, params=None):
        if params is not None:
            self.__init__(params)
        """uploads all imgs in in_array to the s3 bucket"""
        print(" V Uploading PNGs to {}...".format(bucket), flush=True)
        # sleep(0.1)
        self.empty_the_bucket()
        self.__save_times()
        self.__upload_files()



    def empty_the_bucket(self):
        print("\t* Emptying Bucket...", end='')
        bucket.objects.all().delete()
        print("Done!")

    def get_file_list(self, force=False):
        # to_upload = self.params.local_imgs_paths()

        if self.to_upload is None or force:
            self.to_upload = [file for file in self.params.local_imgs_paths() if ("_orig" not in file)]
            # for file in self.to_upload:
            #     file.replace("synoptic", "gilly")

            if self.params.do_orig and False:
                for file in os.listdir(self.params.orig_directory):
                    self.to_upload.append(os.path.join(self.params.orig_directory, file))

            if self.params.do_compare and False:
                comp_dir = self.params.orig_directory.replace('orig', 'compare')
                for file in os.listdir(comp_dir):
                    self.to_upload.append(os.path.join(comp_dir, file))

            # print(" V Uploading Files")
        self.pbar = tqdm(self.to_upload, desc="\r\t* Uploading Files", ncols=120)
        # [print(x) for x in self.to_upload]
        return self.to_upload, self.pbar

    def __upload_files(self):

        to_upload, pbar = self.get_file_list()
        if self.params.multi_pool is not None:
            results = self.params.multi_pool.imap(self.do_upload, to_upload)
            for res in results:
                pbar.update()
                self.ii += 1
        else:
            self.upload_serial(to_upload, pbar)
        pbar.close()

        print(" ^ Success! Uploaded {} PNGs\n".format(len(self.params.local_imgs_paths())))

    def upload_serial(self, to_upload=None, pbar=None):

        if to_upload is None:
            to_upload, pbar = self.get_file_list()

        for upload in to_upload:
            self.do_upload(upload)
            pbar.update()
            self.ii += 1

    @staticmethod
    def do_upload(root_path):
            smallPath, rtPath, smallAWSpath, bigAWSpath = make_thumbs(root_path)

            # Upload large File
            bucket.upload_file(root_path, bigAWSpath, ExtraArgs=png_args)

            # Upload Thumbnail
            bucket.upload_file(smallPath, smallAWSpath, ExtraArgs=png_args)


    def __save_times(self):
        """Saves the Time file to S3 so we know when images were taken"""
        print("\t* Uploading Time File...",end='', flush=True)
        path = self.params.time_path()
        path2 = path.replace(".txt", "_readable.txt")

        # Read in the Input
        frame, wave, t_rec, center, int_time, nm = self.load_this_fits_frame(self.params.local_fits_paths()[0], self.params.master_frame_list_newest)

        # Write the raw output
        # shortened = t_rec.split('.')[0]
        with open(path, "w") as fp:
            fp.write(t_rec)

        tz_list = []
        nzt = self.clean_time_string(t_rec, "NZ"    ).replace("NZDT, ", "NZDT,").replace("NZST, ", "NZST,")
        tz_list.append(nzt)
        tz_list.append(self.clean_time_string(t_rec, 'Japan'   ))
        # tz_list.append(self.clean_time_string(t_rec, "Iran"    ))
        tz_list.append(self.clean_time_string(t_rec, "EET"    ).replace("EEST, ", "EEST,"))
        tz_list.append("       ~*~")

        # tz_list.append(self.clean_time_string(t_rec, "Europe/Berlin"    ))
        tz_list.append(self.clean_time_string(t_rec, None           ))
        tz_list.append("       ~*~")

        tz_list.append(self.clean_time_string(t_rec, "US/Eastern"   ))
        tz_list.append(self.clean_time_string(t_rec, "US/Central"  ))
        tz_list.append(self.clean_time_string(t_rec, "US/Mountain"  ))
        tz_list.append(self.clean_time_string(t_rec, "US/Pacific"   ))
        tz_list.append(self.clean_time_string(t_rec, "US/Hawaii"    ))

        with open(path2, "w") as fp:
            for item in tz_list:
                fp.write(item)
                fp.write("\n")

        # up_path_1 = "sunback_images/{}".format(os.path.basename(path))
        # up_path_2 = "sunback_images/{}".format(os.path.basename(path2))
        up_path_1 = os.path.basename(path)
        up_path_2 = os.path.basename(path2)
        bucket.upload_file(path, up_path_1, ExtraArgs=txt_args)
        bucket.upload_file(path2,up_path_2, ExtraArgs=txt_args)

        print("Done! ", flush=True)

    # def put_ultimate(self):
    #     """uploads all imgs in in_array to the s3 bucket"""
    #     print("   Uploading files to {}...".format(bucket), flush=True)
    #     sleep(0.1)
    #     for local, remote in tqdm(self.params.local_imgs_paths()):
    #
    #         # Upload file
    #         bucket.upload_file(local, remote, ExtraArgs=S3_UPLOAD_ARGS)
    #
    #     self.__save_times()
    #     print("  Success! Uploaded {} files\n".format(len(self.params.local_imgs_paths())))