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

from datetime import datetime, timezone

from sunback.utils.array_util import get_thumblinks, make_thumbs
from sunback.putter.serve_keys import serve_id_for_local_png, s3_img_key, s3_thumb_key

THUMB_PX = 256  # served thumbnail size (page card)

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
        # obstime stamps the 1k stills so the Lambda can order the video queue.
        # Upload time ~= observation time (reducer runs right after NRT publish).
        self.obstime = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.__upload_files()
        self.__upload_tscan_video()
        self.__save_times()

    def __upload_tscan_video(self):
        """Upload the DEM temperature-scan video to a stable key for the DEM card.

        This is a pre-rendered scan over temperature (not a 48h timelapse), so it
        bypasses the Lambda and is uploaded directly, overwriting each run.
        """
        import glob
        roots = [self.params.imgs_top_directory(), self.params.base_directory()]
        found = None
        for root in roots:
            if not root:
                continue
            for pat in ("a_temp_video_small.mp4", "a_temp_video.mp4"):
                hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
                if hits:
                    found = hits[0]
                    break
            if found:
                break
        if not found:
            print("\t* No temperature-scan video found; skipping.")
            return
        bucket.upload_file(found, "video/rhef_tscan.mp4", ExtraArgs=video_args)
        print(f"\t* Uploaded temperature-scan video -> video/rhef_tscan.mp4")

    def empty_the_bucket(self):
        print("\t* Emptying Bucket...", end='')
        bucket.objects.all().delete()
        print("Done!")

    def get_file_list(self, force=False):
        if self.to_upload is None or force:
            # Only the served PNG stills. Videos are built by the Lambda, not here,
            # so the .mp4 is no longer collected/uploaded.
            self.to_upload = [
                f for f in self.params.local_imgs_paths()
                if "_orig" not in f and serve_id_for_local_png(f) is not None
            ]

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

    def do_upload(self, root_path):
        """Upload one served still as 1k/rhef_<id>_1k.png + a 256² thumb.

        The 1k still upload is what fires the Lambda video-builder; obstime
        metadata lets the Lambda order the 48h frame queue.
        """
        product_id = serve_id_for_local_png(root_path)
        if product_id is None:
            return  # not a served product (UV-only channel, DEM, alt composite, ...)

        meta = {"obstime": getattr(self, "obstime", "")}
        img_args = copy(png_args)
        img_args["Metadata"] = meta

        # full-res 1k still
        bucket.upload_file(root_path, s3_img_key(product_id), ExtraArgs=img_args)

        # 256² thumbnail (square 1024² source -> direct resize)
        img = cv2.imread(root_path, cv2.IMREAD_UNCHANGED)
        thumb_path = os.path.join(os.path.dirname(root_path),
                                  f".thumb_{product_id}.png")
        cv2.imwrite(thumb_path, cv2.resize(img, (THUMB_PX, THUMB_PX),
                                           interpolation=cv2.INTER_AREA))
        bucket.upload_file(thumb_path, s3_thumb_key(product_id), ExtraArgs=png_args)

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