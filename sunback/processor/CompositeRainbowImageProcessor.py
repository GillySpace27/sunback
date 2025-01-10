import cv2
import os
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from pathlib import Path
import logging
from sunback.processor.Processor import Processor
from sunback.processor.ImageProcessorCV import ImageProcessorCV
logging.basicConfig(level=logging.DEBUG)


class RainbowRGBImageProcessor(ImageProcessorCV):
    """Processor for generating RGB images from FITS files."""
    complete = False
    filt_name = "RGB Image Writer"
    description = "Generate RGB versions of the images"
    progress_verb = "Creating"
    progress_unit = "Images"
    finished_verb = "Written to Disk"
    out_name = "rgb"
    can_do_parallel = True

    def __init__(self, params=None, quick=None, rp=False, rgb1=("0171", "0193", "0211"), rgb2=("0094", "0131", "0335"), rgb3=("1600", "1700", "0304")):
        super().__init__(params, quick)
        self.params = params
        self.rgb_channels = [rgb1, None, None]
        self.good_paths = {}
        self.missing_files = []
        self.missing_counts = 0
        self.params.rp = rp

    def process_fits_series(self):
        return self.do_work()

    def do_work(self):
        if self.complete:
            raise StopIteration

        self.collect_fits_paths()
        if not self.good_paths:
            logging.error("No valid FITS files found. Exiting.")
            return

        self.create_rgb_images()
        self.complete = True
        self.report_missing_files()

    def collect_fits_paths(self):
        """Collect paths to FITS files from the rainbow directory."""
        base_dir = Path(self.params.base_directory()).parent
        fits_dir = base_dir / "rainbow" / "imgs" / "fits"
        logging.debug(f"Checking rainbow mode directory: {fits_dir}")

        if fits_dir.exists():
            fits_files = sorted(fits_dir.glob("AIAsynoptic*.fits"))
            for file_path in fits_files:
                wavelength = file_path.stem.split("AIAsynoptic")[1]
                if self.is_valid_fits_file(file_path, "rhef"):
                    logging.debug(f"Valid FITS file: {file_path}")
                    self.good_paths[wavelength] = str(file_path)
                else:
                    logging.warning(f"Invalid FITS file: {file_path}")
                    self.missing_files.append(str(file_path))
                    self.missing_counts += 1
        else:
            logging.warning(f"Directory does not exist: {fits_dir}")

    def is_valid_fits_file(self, file_path, hdu_name):
        """Check if the FITS file contains the specified HDU."""
        try:
            with fits.open(file_path) as hdul:
                for hdu in hdul:
                    if hdu.name.casefold() == hdu_name.casefold():
                        if hdu.data is None or hdu.data.size == 0 or hdu.data.shape != (1024, 1024):
                            logging.warning(f"HDU '{hdu_name}' in file {file_path} is empty or has an unexpected shape.")
                            return False
                        return True
        except Exception as e:
            logging.error(f"Error checking FITS file {file_path}: {e}")
        return False

    def create_rgb_images(self):
        """Create RGB images and save them in the output folder."""
        output_folder = Path(self.params.base_directory()).parent / "rainbow" / "imgs" / "mod"
        output_folder.mkdir(parents=True, exist_ok=True)

        try:
            loaded_data = {}
            for wavelength, file_path in self.good_paths.items():
                loaded_data[wavelength] = self.load_fits_data(file_path)

            if any(data is None for data in loaded_data.values()):
                logging.warning("One or more FITS files could not be loaded. Skipping RGB creation.")
                return

            # Create and save the three RGB images
            for index, channels in enumerate(self.rgb_channels):
                if channels is not None:
                    try:
                        data_R = np.flipud(loaded_data[channels[0]])
                        data_G = np.flipud(loaded_data[channels[1]])
                        data_B = np.flipud(loaded_data[channels[2]])

                        if "RHEF" in self.params.png_frame_name:
                            data_R, _ = self.do_norm_stretch(data_R, self.params.png_frame_name[0])
                            data_G, _ = self.do_norm_stretch(data_G, self.params.png_frame_name[0])
                            data_B, _ = self.do_norm_stretch(data_B, self.params.png_frame_name[0])

                        try:
                            data_R = self.label_plot(data_R)
                            data_G = self.label_plot(data_G)
                            data_B = self.label_plot(data_B)
                        except (ValueError, AttributeError) as e:
                            print(110, e)

                        # Create the RGB image
                        img_rgb = self.make_unscaled_rgb(data_R, data_G, data_B)
                        img_8bit = img_rgb.astype(np.uint8)

                        # Save the output image
                        output_path = output_folder / f"BGR_{channels[0]}_{channels[1]}_{channels[2]}_{self.params.png_frame_name[0]}.png"
                        cv2.imwrite(str(output_path), img_8bit)
                        logging.debug(f"Saved to {output_path} !")
                    except IndexError:
                        logging.error(f"Error processing channel combination {channels}. Skipping.")
                        continue
        except Exception as e:
            logging.error(f"Error processing rainbow frames: {e}")

    def report_missing_files(self):
        """Report the missing FITS files and list them."""
        if self.missing_counts > 0:
            logging.info(f"Missing files: {self.missing_counts} out of {self.missing_counts + len(self.good_paths)}")
            for file_path in self.missing_files[:10]:  # Limit to first 10 files for logging
                logging.info(f"  - {file_path}")
            if self.missing_counts > 10:
                logging.info(f"  ...and {self.missing_counts - 10} more files.")
        else:
            logging.debug("No missing files were detected.")

    def load_fits_data(self, file_path, hdu_name_or_index="rhef"):
        """Load data from a specified HDU of a FITS file by name or index."""
        logging.debug(f"Loading FITS file: {file_path}")
        if not os.path.exists(file_path):
            logging.error(f"FITS file not found: {file_path}")
            return None
        try:
            with fits.open(file_path) as hdul:
                logging.debug(f"Opened FITS file: {file_path}")
                if isinstance(hdu_name_or_index, str):
                    for hdu in hdul:
                        if hdu.name.casefold() == hdu_name_or_index.casefold():
                            if hdu.data is None or hdu.data.size == 0:
                                logging.error(f"HDU '{hdu_name_or_index}' in file {file_path} is empty or has an unexpected shape.")
                                return None
                            logging.debug(f"Loaded data from HDU '{hdu_name_or_index}' in file {file_path}")
                            return hdu.data
                elif isinstance(hdu_name_or_index, int):
                    if -len(hdul) <= hdu_name_or_index < len(hdul):
                        hdu = hdul[hdu_name_or_index]
                        if hdu.data is None or hdu.data.size == 0 or hdu.data.shape != (1024, 1024):
                            logging.error(f"HDU index {hdu_name_or_index} in file {file_path} is empty or has an unexpected shape.")
                            return None
                        logging.debug(f"Loaded data from HDU index {hdu_name_or_index} in file {file_path}")
                        return hdu.data
                else:
                    logging.error("HDU identifier must be either an integer index or a string name.")
        except Exception as e:
            logging.error(f"Error loading FITS file {file_path}: {e}")
        return None


    def make_unscaled_rgb(self, image_r, image_g, image_b):
        """Create an RGB image from three channels."""
        def to_int8(image):
            if image.dtype == np.int8:
                return image
            elif np.issubdtype(image.dtype, np.floating):
                if not 0 <= np.nanmin(image) <= 1 or not 0 <= np.nanmax(image) <= 1:
                    pass
                return (image * 255).astype(np.int8)
            else:
                raise ValueError("Input images must be of dtype int8 or convertible to int8 (float values between 0 and 1).")



        # Convert to int8 if necessary
        image_r = to_int8(image_r)
        image_g = to_int8(image_g)
        image_b = to_int8(image_b)

        # Stack to form RGB
        rgb = np.stack((image_r, image_g, image_b), axis=-1)
        return rgb
