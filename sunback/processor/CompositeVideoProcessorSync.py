import cv2
import os
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from collections import defaultdict
from pathlib import Path
import logging
import re
import datetime
from sunback.processor.Processor import Processor

logging.basicConfig(level=logging.DEBUG)

class RGBImageProcessor(Processor):
    """Processor for generating RGB images from FITS files."""
    complete = False
    filt_name = "RGB Image Writer"
    description = "Turn all the FITS files into PNG files"
    progress_verb = "Creating"
    progress_unit = "Images"
    finished_verb = "Written to Disk"
    out_name = "rgb"
    can_do_parallel = True

    def __init__(self, params=None, quick=None, iR=211, iG=193, iB=171, rp=None):
        super().__init__(params, quick, rp)
        self.params = params
        self.iR = iR
        self.iG = iG
        self.iB = iB
        self.good_paths = defaultdict(list)
        self.incomplete = []
        self.missing_files = defaultdict(list)  # To track missing files for each wavelength
        self.missing_counts = defaultdict(int)  # To count missing files for each wavelength

    def do_work(self):
        if self.complete:
            raise StopIteration
        self.collect_fits_paths()
        self.sync_frames_by_timestamp()
        self.create_rgb_images()
        self.complete = True
        self.report_missing_files()

    def collect_fits_paths(self):
        """Collect paths to FITS files from each wavelength's 'fits' directory."""
        base_dir = Path(self.params.base_directory()).parent
        logging.debug(f"Looking for FITS files in base directory: {base_dir}")

        for wavelength in [self.iR, self.iG, self.iB]:
            fits_dir = base_dir / f"{wavelength:04d}" / "imgs" / "fits"
            logging.debug(f"Checking FITS directory: {fits_dir}")

            if fits_dir.exists():
                fits_files = sorted(fits_dir.glob("*.fits"))
                logging.info(f"Found {len(fits_files)} FITS files in {fits_dir}")

                # Store the paths with corresponding timestamps only if the "rhef" HDU is present
                for file_path in fits_files:
                    timestamp = self.extract_timestamp(file_path.name)
                    if self.is_valid_fits_file(file_path, "rhef"):
                        self.good_paths[wavelength].append((timestamp, str(file_path)))
                    else:
                        # Record the missing file
                        self.missing_files[wavelength].append(str(file_path))
                        self.missing_counts[wavelength] += 1
            else:
                logging.warning(f"Directory does not exist: {fits_dir}")

    def is_valid_fits_file(self, file_path, hdu_name):
        """Check if the FITS file contains the specified HDU."""
        try:
            with fits.open(file_path) as hdul:
                for hdu in hdul:
                    if hdu.name.casefold() == hdu_name.casefold():
                        return True
        except Exception as e:
            logging.error(f"Error checking FITS file {file_path}: {e}")
        return False

    def extract_timestamp(self, filename):
        """Extract timestamp from the FITS filename using a regex pattern."""
        # Try matching 'YYYYMMDD_HHMMSS' format
        match = re.search(r'(\d{8}_\d{6})', filename)
        if match:
            return datetime.datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')

        # Fallback to 'YYYYMMDD_HHMM' format
        match = re.search(r'(\d{8}_\d{4})', filename)
        if match:
            return datetime.datetime.strptime(match.group(1), '%Y%m%d_%H%M')

        # Raise an error if no timestamp found
        raise ValueError(f"Could not extract timestamp from {filename}")

    def sync_frames_by_timestamp(self):
        """Sync frames across wavelengths by matching timestamps."""
        # Collect timestamps for each wavelength
        all_timestamps = {
            wavelength: {ts for ts, _ in self.good_paths[wavelength]}
            for wavelength in [self.iR, self.iG, self.iB]
        }

        # Find common timestamps across all wavelengths
        common_timestamps = set.intersection(
            all_timestamps[self.iR], all_timestamps[self.iG], all_timestamps[self.iB]
        )

        # Filter paths to keep only synced timestamps
        for wavelength in [self.iR, self.iG, self.iB]:
            self.good_paths[wavelength] = [
                (ts, path) for ts, path in self.good_paths[wavelength] if ts in common_timestamps
            ]

        logging.info(f"Synced {len(common_timestamps)} frames across wavelengths")

    def report_missing_files(self):
        """Report the wavelength with the most missing files and list them."""
        # Determine which wavelength has the most missing files
        most_lossy_wavelength = max(self.missing_counts, key=self.missing_counts.get, default=None)
        if most_lossy_wavelength is not None:
            logging.info(f"Wavelength {most_lossy_wavelength} had the most missing files: {self.missing_counts[most_lossy_wavelength]} out of {len(self.good_paths[most_lossy_wavelength]) + self.missing_counts[most_lossy_wavelength]}")
        else:
            logging.info("No missing files were detected.")

        # Log details of all missing files if there are any
        for wavelength, files in self.missing_files.items():
            missing_count = len(files)
            total_count = missing_count + len(self.good_paths[wavelength])
            percentage_missing = (missing_count / total_count) * 100 if total_count > 0 else 0
            logging.info(f"Wavelength {wavelength}: {missing_count}/{total_count} files missing ({percentage_missing:.2f}%)")
            if missing_count > 0:
                for file_path in files[:10]:  # Limit to first 10 files for logging
                    logging.info(f"  - {file_path}")
                if missing_count > 10:
                    logging.info(f"  ...and {missing_count - 10} more files.")

    def load_fits_data(self, file_path, hdu_name_or_index="rhef"):
        """Load data from a specified HDU of a FITS file by name or index."""
        if not os.path.exists(file_path):
            logging.error(f"FITS file not found: {file_path}")
            return None
        try:
            with fits.open(file_path) as hdul:
                if isinstance(hdu_name_or_index, str):
                    for hdu in hdul:
                        if hdu.name.casefold() == hdu_name_or_index.casefold():
                            if hdu.data is None or hdu.data.size == 0:
                                logging.error(f"HDU 'rhef' in file {file_path} is empty.")
                                return None
                            return hdu.data
                elif isinstance(hdu_name_or_index, int):
                    if -len(hdul) <= hdu_name_or_index < len(hdul):
                        hdu = hdul[hdu_name_or_index]
                        if hdu.data is None or hdu.data.size == 0:
                            logging.error(f"HDU index {hdu_name_or_index} in file {file_path} is empty.")
                            return None
                        return hdu.data
                else:
                    logging.error("HDU identifier must be either an integer index or a string name.")
        except Exception as e:
            logging.error(f"Error loading FITS file {file_path}: {e}")
        return None

    def create_rgb_images(self):
        """Create RGB images and save them in the output folder."""
        output_folder_name = f"{self.iR}_{self.iG}_{self.iB}_RGB"
        output_folder = Path(self.params.base_directory()).parent / output_folder_name / "imgs"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Ensure the timestamps across the channels are synchronized
        num_frames = len(self.good_paths[self.iR])  # Assumes the channels are synced

        for i in tqdm(range(num_frames), desc="Creating RGB images", unit="images"):
            try:
                # Get the synchronized timestamp and file paths for R, G, B
                timestamp_R, file_path_R = self.good_paths[self.iR][i]
                timestamp_G, file_path_G = self.good_paths[self.iG][i]
                timestamp_B, file_path_B = self.good_paths[self.iB][i]

                # Load the actual data from the file paths
                data_R = self.load_fits_data(file_path_R)
                data_G = self.load_fits_data(file_path_G)
                data_B = self.load_fits_data(file_path_B)

                # Skip if any data is missing or invalid
                if data_R is None or data_G is None or data_B is None:
                    logging.warning(f"Missing or invalid data for frame {i}. Skipping RGB creation.")
                    continue

                # Ensure data has the expected shape (e.g., 1024x1024)
                expected_shape = (1024, 1024)
                if data_R.shape != expected_shape or data_G.shape != expected_shape or data_B.shape != expected_shape:
                    logging.error(f"Data shape mismatch for frame {i}. Expected {expected_shape}, got R: {data_R.shape}, G: {data_G.shape}, B: {data_B.shape}. Skipping.")
                    continue

                # Create timestamp string for overlay
                timestamp_str = timestamp_R.strftime('%Y-%m-%d %H:%M:%S')

                # Create the RGB image
                img_rgb = self.make_unscaled_rgb(data_R, data_G, data_B)
                img_8bit = (img_rgb).astype(np.uint8)

                # Add timestamp text to the image
                cv2.putText(img_8bit, timestamp_str, (10, img_8bit.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Save the output image
                output_path = output_folder / f"RGB_{i:04d}.png"
                cv2.imwrite(str(output_path), img_8bit)

            except Exception as e:
                logging.error(f"Error processing frame {i}: {e}")

    @staticmethod
    def make_unscaled_rgb(image_r, image_g, image_b, filename=None):
        """Create an RGB image from three channels."""
        def to_int8(image):
            if image.dtype == np.int8:
                return image
            elif np.issubdtype(image.dtype, np.floating) and 0 <= np.nanmin(image) <= 1 and 0 <= np.nanmax(image) <= 1:
                return (image * 255).astype(np.int8)
            else:
                raise ValueError("Input images must be of dtype int8 or convertible to int8 (float values between 0 and 1).")

        # Convert to int8 if necessary
        image_r = to_int8(image_r)
        image_g = to_int8(image_g)
        image_b = to_int8(image_b)

        # Stack to form RGB
        rgb = np.stack((image_r, image_g, image_b), axis=-1)

        # Optionally save to file
        if filename:
            import matplotlib.image
            matplotlib.image.imsave(filename, rgb, origin="lower")

        return rgb