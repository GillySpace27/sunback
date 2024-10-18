import cv2
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from collections import defaultdict
from pathlib import Path
import logging
logging.basicConfig(level=logging.DEBUG)
import re
import datetime
from src.processor.Processor import Processor
from src.processor.ImageProcessor import ImageProcessor
from src.processor.VideoProcessor import VideoProcessor

class RGBImageProcessor(Processor):
    """Processor for generating RGB images from FITS files."""
    complete = False
    filt_name = "RGB Image Writer"
    description = "Turn all the FITS files into PNG files"
    progress_verb = "Creating"
    progress_unit = "Images"
    finished_verb = "Written to Disk"
    out_name = "rgb"

    def __init__(self, params=None, quick=None, iR=171, iG=193, iB=211, rp=None):
        super().__init__(params, quick, rp)  # Call the parent constructor

        self.params = params
        self.iR = iR
        self.iG = iG
        self.iB = iB
        self.good_paths = defaultdict(list)

    def do_work(self):
        if self.complete:
            raise StopIteration
        self.collect_fits_paths()
        self.create_rgb_images()
        self.complete = True

    def collect_fits_paths(self):
        """Collect paths to FITS files from each wavelength's 'fits' directory."""
        base_dir = Path(self.params.base_directory()).parent  # Use params to get base directory
        logging.debug(f"Looking for FITS files in base directory: {base_dir}")

        for wavelength in [self.iR, self.iG, self.iB]:
            fits_dir = base_dir / f"{wavelength:04d}" / "imgs" / "fits"
            logging.debug(f"Checking FITS directory: {fits_dir}")

            if fits_dir.exists():
                fits_files = sorted(fits_dir.glob("*.fits"))
                logging.info(f"Found {len(fits_files)} FITS files in {fits_dir}")

                # Store the paths without any syncing or timestamp logic
                self.good_paths[wavelength] = [str(file_path) for file_path in fits_files]
            else:
                logging.warning(f"Directory does not exist: {fits_dir}")

    def create_rgb_images(self):
        """Create RGB images and save them in the output folder."""
        output_folder_name = f"{self.iR}_{self.iG}_{self.iB}_RGB"
        output_folder = Path(self.params.base_directory()).parent / output_folder_name / "imgs"
        output_folder.mkdir(parents=True, exist_ok=True)

        num_frames = min(
            len(self.good_paths[self.iR]),
            len(self.good_paths[self.iG]),
            len(self.good_paths[self.iB]),
        )

        for i in tqdm(range(num_frames), desc="Creating RGB images", unit="images"):
            try:
                # Load data for each wavelength
                data_R = self.load_fits_data(self.good_paths[self.iR][i])
                data_G = self.load_fits_data(self.good_paths[self.iG][i])
                data_B = self.load_fits_data(self.good_paths[self.iB][i])

                # Check for missing data
                if data_R is None or data_G is None or data_B is None:
                    logging.error(
                        "Missing data for wavelengths: "
                        f"{'R' if data_R is None else ''} "
                        f"{'G' if data_G is None else ''} "
                        f"{'B' if data_B is None else ''}"
                    )
                    continue

                # Create RGB composite image
                img_rgb = make_lupton_rgb(data_R, data_G, data_B, Q=0, stretch=1)
                img_8bit = (img_rgb).astype(np.uint8)  # Convert to 8-bit

                # Save the RGB image
                output_path = output_folder / f"RGB_{i:04d}.png"
                cv2.imwrite(str(output_path), img_8bit)

            except Exception as e:
                logging.error(f"Error processing frame {i}: {e}")

    def load_fits_data(self, file_path):
        """Load data from the last HDU of a FITS file."""
        with fits.open(file_path) as hdul:
            return hdul[-1].data


class RGBVideoWriterProcessor(Processor):
    """Processor for creating a video from generated RGB images."""
    filt_name = "RGB Video Writer"
    description = "Turn all the png files into a video"
    progress_verb = "Encoding"
    progress_unit = "Images"
    finished_verb = "Written to Disk"
    out_name = "png"
    complete = False

    def __init__(self, params=None, quick=None, iR=171, iG=193, iB=211, fps=30, rp=False):
        super().__init__(params, quick, rp)  # Call the parent constructor
        self.iR = iR
        self.iG = iG
        self.iB = iB
        self.fps = fps

    def process(self, params):
        if self.complete:
            raise StopIteration
        self.create_video_from_images()
        self.complete = True

    def create_video_from_images(self):
        """Create a video from RGB images saved in the output folder."""
        output_folder_name = f"{self.iR}_{self.iG}_{self.iB}_RGB"
        output_folder = Path(self.params.base_directory()).parent / output_folder_name / "imgs"
        video_path = output_folder.parent / "composite_video.avi"
        import os.path
        if os.path.exists(video_path):
            raise StopIteration
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI

        # Get the first image to determine the dimensions
        img_files = sorted(output_folder.glob("*.png"))
        if not img_files:
            logging.error("No RGB images found to create video.")
            return

        first_img = cv2.imread(str(img_files[0]))
        height, width = first_img.shape[:2]

        # Initialize the VideoWriter
        video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))

        for img_file in tqdm(img_files, desc=self.progress_verb+" "+self.progress_unit, unit="frames"):
            img = cv2.imread(str(img_file))
            video_writer.write(img)

        # Release the VideoWriter
        video_writer.release()
        logging.info(f"Video saved as {video_path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parameters would be initialized appropriately in your actual code
    params = None  # Replace with actual parameter object

    # Create RGB images
    rgb_processor = RGBImageProcessor(params)
    rgb_processor.collect_fits_paths()
    rgb_processor.create_rgb_images()

    # Create video from RGB images
    video_processor = RGBVideoWriterProcessor(params)
    video_processor.create_video_from_images()