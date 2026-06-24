from os.path import split
from os import makedirs
from time import time
from tqdm import tqdm
from sunback.putter.Putter import Putter
import boto3
import cv2
import os
from copy import copy
from time import sleep

from sunback.utils.array_util import get_thumblinks, make_thumbs

S3_UPLOAD_ARGS = {'ACL': 'public-read', "ContentDisposition": "inline"}

txt_args = copy(S3_UPLOAD_ARGS)
txt_args["ContentType"] = "text/plain"

png_args = copy(S3_UPLOAD_ARGS)
png_args["ContentType"] = "image/png"

video_args = copy(S3_UPLOAD_ARGS)
video_args["ContentType"] = "video/mp4"

s3 = boto3.resource('s3')
bucket_name = 'the-sun-now'
bucket = s3.Bucket(bucket_name)
s3_client = boto3.client('s3')

class AwsPutter(Putter):
    filt_name = "AWSputter"
    description = "Upload Images to AWS {}".format(bucket_name)
    progress_verb = "Uploaded"
    progress_unit = "Images"

    def __init__(self, params=None, quick=False, rp=None, in_name=None):
        super().__init__(params, quick, rp, in_name)
        self.ii = 0
        self.pbar = None
        self.to_upload = None

    def put(self, params=None):
        if params is not None:
            self.__init__(params)
        print(" V Uploading PNGs to {}...".format(bucket), flush=True)
        # NOTE: do NOT empty the bucket. The Lambda video-builder maintains the
        # frames/ queue and video/ outputs there; wiping would destroy the 48h
        # sliding window every run. The reducer only overwrites its own keys.
        # self.empty_the_bucket()   # <-- intentionally disabled (see spec)
        self.__upload_files()
        self.__save_times()

    def empty_the_bucket(self):
        print("\t* Emptying Bucket...", end='')
        bucket.objects.all().delete()
        print("Done!")

    def get_file_list(self, force=False):
        if self.to_upload is None or force:
            self.to_upload = [file for file in self.params.local_imgs_paths() if ("_orig" not in file)]

            try:
                local_paths = self.params.local_imgs_paths()
                if local_paths:
                    out_dir = os.path.dirname(local_paths[0])
                    mov_files = [x for x in os.listdir(out_dir) if x.lower().endswith(".mp4")]
                    if mov_files:
                        mov = mov_files[0]
                        self.to_upload.append(os.path.join(out_dir, mov))
            except Exception as e:
                print(f"Warning: Could not add .mp4 file from directory due to error: {e}")

        self.pbar = tqdm(self.to_upload, desc="\r\t* Uploading Files", ncols=120)
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
        original_path = root_path

        # If it's a video, generate and upload thumbnail only to thumbs
        if root_path.lower().endswith(".mp4"):
            cap = cv2.VideoCapture(root_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Warning: Could not extract frame from {root_path}")
                return

            # Create path for thumbnail in thumbs directory
            thumb_rel_path = os.path.basename(root_path).replace(".mp4", "_frame_for_thumb.png")
            temp_image_path = os.path.join(os.path.dirname(root_path), "thumbs", thumb_rel_path)
            os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
            cv2.imwrite(temp_image_path, frame)

            # Upload .mp4 to renders/
            video_key = os.path.join("renders", os.path.basename(original_path))
            bucket.upload_file(original_path, video_key, ExtraArgs=video_args)

            # Create & upload thumbs from extracted frame only
            smallPath, rtPath, smallAWSpath, bigAWSpath = make_thumbs(temp_image_path)
            bucket.upload_file(smallPath, smallAWSpath, ExtraArgs=png_args)

        else:
            # Normal image case
            smallPath, rtPath, smallAWSpath, bigAWSpath = make_thumbs(root_path)
            bucket.upload_file(root_path, bigAWSpath, ExtraArgs=png_args)
            bucket.upload_file(smallPath, smallAWSpath, ExtraArgs=png_args)

    def __save_times(self):
        print("\t* Uploading Time File...", end='', flush=True)
        path = self.params.time_path()
        path2 = path.replace(".txt", "_readable.txt")

        frame, wave, t_rec, center, int_time, nm = self.load_this_fits_frame(self.params.local_fits_paths()[0], -1)

        with open(path, "w") as fp:
            fp.write(t_rec)

        tz_list = []
        nzt = self.clean_time_string(t_rec, "NZ").replace("NZDT, ", "NZDT,").replace("NZST, ", "NZST,")
        tz_list.append(nzt)
        tz_list.append(self.clean_time_string(t_rec, 'Japan'))
        tz_list.append(self.clean_time_string(t_rec, "EET").replace("EEST, ", "EEST,"))
        tz_list.append("       ~*~")
        tz_list.append(self.clean_time_string(t_rec, None))
        tz_list.append("       ~*~")
        tz_list.append(self.clean_time_string(t_rec, "US/Eastern"))
        tz_list.append(self.clean_time_string(t_rec, "US/Central"))
        tz_list.append(self.clean_time_string(t_rec, "US/Mountain"))
        tz_list.append(self.clean_time_string(t_rec, "US/Pacific"))
        tz_list.append(self.clean_time_string(t_rec, "US/Hawaii"))

        with open(path2, "w") as fp:
            for item in tz_list:
                fp.write(item + "\n")

        bucket.upload_file(path, os.path.basename(path), ExtraArgs=txt_args)
        bucket.upload_file(path2, os.path.basename(path2), ExtraArgs=txt_args)
        print("Done! ", flush=True)