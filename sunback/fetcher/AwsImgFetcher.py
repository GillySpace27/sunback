import sys
import os
from os.path import join, split
from tqdm import tqdm
import boto3

from sunback.fetcher.Fetcher import Fetcher


class AwsImgFetcher(Fetcher):
    """
    Fetch images from an S3 bucket. Downloads any valid PNG files from
    the specified prefix, placing them into self.params.imgs_top_directory().
    """
    description = "Get imgs from the Amazon S3 Bucket"
    filt_name = "AWS Image Fetcher"
    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.load(params)
        # Set up S3 resources
        self.s3_resource = boto3.resource('s3')

        # Use user-provided bucket/prefix if they exist. Otherwise, hardcode or pick defaults.
        self.bucket_name = getattr(self.params, 's3_bucket_name', 'the-sun-now')
        self.prefix      = getattr(self.params, 's3_prefix',       'renders/')

        self.my_bucket = self.s3_resource.Bucket(self.bucket_name)
        self.n_obj = 0  # Will keep track of how many valid objects we download
        self.download_dir = self.params.imgs_top_directory() if self.params else "./downloaded"

        # This sets up the object filter that we'll iterate over
        self.objects = self.my_bucket.objects.filter(Prefix=self.prefix)

    def fetch(self, params=None):
        """
        Download all valid PNG files from the S3 bucket and store them
        in the download directory (self.params.imgs_top_directory()).
        """
        # If the user called fetch() with new params, update them
        if params is not None:
            self.params = params
            self.__init__(params=params)  # Re-init so that we pick up any new s3_bucket_name/prefix

        # Ensure the download directory exists
        os.makedirs(self.download_dir, exist_ok=True)

        print(f"   Downloading PNGs from '{self.bucket_name}/{self.prefix}' to {self.download_dir}", flush=True)
        sys.stdout.flush()

        valid_keys = self._filter_objects()
        if not valid_keys:
            print("   No files found matching the criteria.")
            return

        # Use tqdm to show progress
        for obj in tqdm(valid_keys, desc="   Downloading Files", ncols=100):
            self._grab_obj(obj)

        # Optionally load into memory or do any post-download steps
        self.load()

        # You could do more robust checks here, but for demonstration:
        print(f"\n   Downloaded {self.n_obj} files in total.")
        sys.stdout.flush()

    def _filter_objects(self):
        """
        Filter out anything you do *not* want to download (like 'orig', 'archive', 'thumbs', '4500').
        Also handle do_one() if the user only wants a single file.
        Returns a list of S3.ObjectSummary items that pass the filters.
        """
        valid_keys = []
        # wanted_one = self.params.do_one() if hasattr(self.params, 'do_one') else None

        for obj in self.objects:
            # Exclude known unwanted substrings
            if any(x in obj.key for x in ['orig', 'archive', 'thumbs', '4500']):
                continue

            # If do_one() is set, skip anything that doesn't match
            # if wanted_one and wanted_one not in obj.key:
            #     continue

            valid_keys.append(obj)

        return valid_keys

    def _grab_obj(self, obj):
        """
        Download a specific S3 object to local disk, prefixed with "dl_" if you like,
        or place it under the same relative path.
        """
        path, filename = split(obj.key)
        local_filename = f"dl_{filename}"
        local_path = join(self.download_dir, local_filename)

        # Ensure any subdirectories exist if needed (e.g., if you preserve path structure)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download File
        self.my_bucket.download_file(obj.key, local_path)

        self.n_obj += 1
        # If you want to print every file as it downloads:
        # print(f"Downloaded: {obj.key} -> {local_path}")
        # sys.stdout.flush()