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
        return match.group(0) if match else None

    def collect_fits_paths(self, wavelengths):
        """Collect paths to FITS files from each wavelength's 'fits' directory."""
        base_dir = os.path.dirname(self.params.base_directory())
        for wavelength in wavelengths:
            fits_dir = os.path.join(base_dir, f"{wavelength:04d}", "imgs", "fits")
            if os.path.exists(fits_dir):
                fits_files = sorted(
                    [
                        os.path.join(fits_dir, f)
                        for f in os.listdir(fits_dir)
                        if f.endswith(".fits")
                    ]
                )
                for file_path in fits_files:
                    timestamp = self.extract_timestamp(file_path)
                    if timestamp:
                        self.good_paths[timestamp][wavelength] = file_path

        # Only retain timestamps that have all three wavelengths to ensure synchronization
        self.good_paths = {
            timestamp: paths
            for timestamp, paths in self.good_paths.items()
            if len(paths) == 3
        }

    def do_work(self):
        """Main method to execute the composite video generation process."""
        if self.process_done:
            self.skipped += 1
            return

        wavelengths = [self.iR, self.iG, self.iB]  # Use dynamically set wavelengths
        self.collect_fits_paths(wavelengths)  # Collect the FITS paths to process
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
                sample_file = list(self.good_paths.values())[0][self.iR]
                with fits.open(sample_file) as hdul:
                    data = hdul[-1].data
                    self.frame_shape = (data.shape[1], data.shape[0])
            else:
                print("Error: No valid FITS files found to determine frame shape.")
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
        self.final_output_path = os.path.abspath(
            os.path.join(self.params.movs_directory(), "../../", file_name)
        )
        thepath = os.path.dirname(self.final_output_path)
        self.rainbow_path = f"{thepath}/rainbow"
        os.makedirs(self.rainbow_path, exist_ok=True)
        return self.final_output_path

    def run_composite_video_writer(self, video_writer):
        """Generate the composite video file using the synchronized frames."""
        last_valid_data = {
            self.iR: None,
            self.iG: None,
            self.iB: None,
        }  # Store last valid frame data for each wavelength

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
            except Exception as e:
                print(f"Error processing frame for timestamp {timestamp}: {e}")
                self.skipped += 1

        video_writer.release()
        print(
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

        # Determine the target shape for resizing, defaulting to self.frame_shape if sz is not provided
        target_shape = (
            (sz, sz) if sz is not None else (self.frame_shape[0], self.frame_shape[1])
        )

        # Calculate the block size for block_reduce based on the target shape
        blk_reduce = (
            data.shape[0] // target_shape[0],
            data.shape[1] // target_shape[1],
        )

        # Use block_reduce for efficient resizing
        data_resized = block_reduce(data, blk_reduce, np.mean)

        return data_resized
