import os
import re
import cv2
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from skimage.transform import resize
from collections import defaultdict
from src.processor.Processor import Processor
from datetime import datetime
from pathlib import Path
import logging

from datetime import timedelta


class CompositeVideoProcessor(Processor):
    mov_type = "avi"
    filt_name = "Composite Video Writer"
    progress_stem = " *    {}"
    progress_verb = "Writing Composite Movie"
    progress_string = progress_stem.format(progress_verb)
    finished_verb = "Wrote Composite Movie"
    progress_unit = "frames"
    progress_text = progress_string
    process_done = False  # Flag to indicate if the process has completed
    ii = 0

    def __init__(
        self,
        params=None,
        quick=False,
        rp=None,
        fill_missing_frames=False,
        iR=171,
        iG=193,
        iB=211,
    ):
        super().__init__(params, quick, rp)
        self.final_output_path = None
        self.frame_shape = None
        self.good_paths = defaultdict(dict)
        self.skipped = 0
        self.fill_missing_frames = (
            fill_missing_frames  # Control flag for missing frames
        )
        self.iR = iR
        self.iG = iG
        self.iB = iB

    def extract_timestamp(self, filename):
        """Extract and truncate the timestamp from the FITS filename to the nearest hour."""
        match = re.search(r"\d{4}-\d{2}-\d{2}T\d{2}", filename)
        if not match:
            # Try AIA format
            match = re.search(r"AIA(\d{8})_(\d{4})", filename)
            if match:
                date = match.group(1)  # YYYYMMDD part
                time = match.group(2)  # HHMM part
                return datetime.strptime(f"{date}T{time[:2]}", "%Y%m%dT%H%M")
        return datetime.strptime(match.group(0), "%Y-%m-%dT%H%M") if match else None

    @staticmethod
    def round_to_nearest_cadence(timestamp, cadence_minutes=2):
        """Round timestamp to the nearest specified cadence in minutes."""
        cadence = timedelta(minutes=cadence_minutes)
        # Calculate the difference in seconds and round
        seconds_since_epoch = (timestamp - datetime(1970, 1, 1)).total_seconds()
        rounded_seconds = (
            round(seconds_since_epoch / cadence.total_seconds())
            * cadence.total_seconds()
        )
        return datetime(1970, 1, 1) + timedelta(seconds=rounded_seconds)

    def collect_fits_paths(self, wavelengths, cadence_threshold_minutes=2):
        """Collect paths to FITS files from each wavelength's 'fits' directory."""
        base_dir = Path(self.params.base_directory()).parent
        logging.debug(f"Looking for FITS files in base directory: {base_dir}")

        # Step 1: Collect files and timestamps for each wavelength
        for wavelength in wavelengths:
            fits_dir = base_dir / f"{wavelength:04d}" / "imgs" / "fits"
            logging.debug(f"Checking FITS directory: {fits_dir}")

            if fits_dir.exists():
                fits_files = sorted(
                    fits_dir.glob("*.fits")
                )  # Directly glob for .fits files
                logging.info(f"Found {len(fits_files)} FITS files in {fits_dir}")

                for file_path in fits_files:
                    logging.debug(f"Processing file: {file_path}")
                    timestamp = self.extract_timestamp(file_path.name)
                    if timestamp:
                        # Round timestamp to the nearest 2-minute cadence
                        rounded_timestamp = self.round_to_nearest_cadence(
                            timestamp, cadence_threshold_minutes
                        )
                        if rounded_timestamp not in self.good_paths:
                            self.good_paths[rounded_timestamp] = {}
                        self.good_paths[rounded_timestamp][wavelength] = str(file_path)
                    else:
                        logging.warning(f"Timestamp extraction failed for {file_path}")

        if not self.good_paths:
            logging.error("No valid FITS paths were found after collection.")
        else:
            logging.info(f"Collected FITS paths for {len(self.good_paths)} timestamps.")

    # def collect_fits_paths(self, wavelengths, cadence_threshold_minutes=5):
    #     """Collect paths to FITS files from each wavelength's 'fits' directory."""
    #     base_dir = Path(self.params.base_directory()).parent
    #     logging.debug(f"Looking for FITS files in base directory: {base_dir}")

    #     # Step 1: Collect files and timestamps for each wavelength
    #     for wavelength in wavelengths:
    #         fits_dir = base_dir / f"{wavelength:04d}" / "imgs" / "fits"
    #         logging.debug(f"Checking FITS directory: {fits_dir}")

    #         if fits_dir.exists():
    #             fits_files = sorted(
    #                 fits_dir.glob("*.fits")
    #             )  # Directly glob for .fits files
    #             logging.info(f"Found {len(fits_files)} FITS files in {fits_dir}")

    #             for file_path in fits_files:
    #                 logging.debug(f"Processing file: {file_path}")
    #                 timestamp = self.extract_timestamp(file_path.name)
    #                 if timestamp:
    #                     timestamp_str = timestamp.strftime("%Y%m%d%H%M")
    #                     if timestamp_str not in self.good_paths:
    #                         self.good_paths[timestamp_str] = {}
    #                     self.good_paths[timestamp_str][wavelength] = str(file_path)
    #                 else:
    #                     logging.warning(f"Timestamp extraction failed for {file_path}")

    #     if not self.good_paths:
    #         logging.error("No valid FITS paths were found after collection.")
    #     else:
    #         logging.info(f"Collected FITS paths for {len(self.good_paths)} timestamps.")

    def do_work(self):
        """Main method to execute the composite video generation process."""
        if self.process_done:
            self.skipped += 1
            return

        wavelengths = [self.iR, self.iG, self.iB]  # Use dynamically set wavelengths
        logging.info(f"Collecting FITS paths for wavelengths: {wavelengths}")

        # Collect the FITS paths to process
        self.collect_fits_paths(wavelengths)

        # Check if valid paths were collected
        if not self.good_paths:
            logging.error("No valid FITS paths were found after collection.")
            return

        # Log the number of valid timestamps found
        logging.info(f"Found {len(self.good_paths)} valid timestamps with FITS files.")

        video_writer = self.init_writer()
        if video_writer:
            self.run_composite_video_writer(video_writer)
            self.process_done = True  # Mark the process as done

    def init_writer(self):
        """Initialize the video writer."""
        self.build_output_path()

        if not self.frame_shape:
            if len(self.good_paths) > 0:
                # Use the first valid file to set the frame shape
                sample_file = list(self.good_paths.values())[0].get(self.iR)
                if sample_file:
                    with fits.open(sample_file) as hdul:
                        data = hdul[-1].data
                        self.frame_shape = (data.shape[1], data.shape[0])
                else:
                    logging.error("No valid FITS files found to determine frame shape.")
                    return None
            else:
                logging.error("No valid FITS paths to determine frame shape.")
                return None

        return cv2.VideoWriter(
            self.final_output_path,
            cv2.VideoWriter.fourcc("M", "J", "P", "G"),
            self.params.frames_per_second(),
            self.frame_shape,
        )

    def build_output_path(self):
        """Build the path to the composite video."""
        batch_name = self.params.config["name"]
        file_name = f"{batch_name}_composite_video.{self.mov_type}"
        self.final_output_path = Path(self.params.movs_directory()).parent / file_name
        self.rainbow_path = self.final_output_path.parent.parent / "rainbow"
        self.rainbow_path.mkdir(parents=True, exist_ok=True)
        self.final_output_path = self.rainbow_path
        return self.final_output_path

    def run_composite_video_writer(self, video_writer):
        """Generate the composite video file using the synchronized frames."""
        last_valid_data = {self.iR: None, self.iG: None, self.iB: None}

        for timestamp, paths in tqdm(
            self.good_paths.items(), desc=self.progress_text, unit="frames"
        ):
            try:
                # Load data for the current frame from each wavelength
                data_R = (
                    self.load_fits_data(paths.get(self.iR))
                    if self.iR in paths
                    else None
                )
                data_G = (
                    self.load_fits_data(paths.get(self.iG))
                    if self.iG in paths
                    else None
                )
                data_B = (
                    self.load_fits_data(paths.get(self.iB))
                    if self.iB in paths
                    else None
                )

                # Handle missing frames based on the fill_missing_frames flag
                data_R = self.handle_missing_data(data_R, last_valid_data, self.iR)
                data_G = self.handle_missing_data(data_G, last_valid_data, self.iG)
                data_B = self.handle_missing_data(data_B, last_valid_data, self.iB)

                # Normalize and resize the data to match target frame shape
                norm_R = self.normalize_and_resize(data_R)
                norm_G = self.normalize_and_resize(data_G)
                norm_B = self.normalize_and_resize(data_B)

                # Create RGB composite image
                img_rgb = make_lupton_rgb(norm_R, norm_G, norm_B, Q=0, stretch=1)
                img_8bit = img_rgb.astype(np.uint8)  # Convert to 8-bit

                video_writer.write(img_8bit)

                # if os.path.exists(self.final_output_path):
                print(f"Saved to: {self.final_output_path}")

            except Exception as e:
                logging.error(f"Error processing frame for timestamp {timestamp}: {e}")
                self.skipped += 1

        video_writer.release()
        logging.info(
            f" ^    Successfully {self.finished_verb} with {len(self.good_paths) - self.skipped} frames! ({self.skipped} skipped)"
        )

    def handle_missing_data(self, data, last_valid_data, wavelength):
        """Handle missing data by using the last valid data or a placeholder if specified."""
        if data is None:
            if self.fill_missing_frames:
                return np.full(
                    self.frame_shape, 0.25
                )  # Use a neutral placeholder if the data is missing
            else:
                return (
                    last_valid_data[wavelength]
                    if last_valid_data[wavelength] is not None
                    else np.full(self.frame_shape, 0.25)
                )
        else:
            last_valid_data[wavelength] = data  # Update the last valid frame data
            return data

    def load_fits_data(self, file_path):
        """Load data from the last HDU of a FITS file."""
        with fits.open(file_path) as hdul:
            return hdul[-1].data

    def normalize_and_resize(self, data, sz=None):
        """Resize data to match the target shape using block_reduce for efficiency."""
        from skimage.measure import block_reduce

        target_shape = (sz, sz) if sz else self.frame_shape
        blk_reduce = (
            data.shape[0] // target_shape[0],
            data.shape[1] // target_shape[1],
        )

        data_resized = block_reduce(data, blk_reduce, np.mean)
        return data_resized
