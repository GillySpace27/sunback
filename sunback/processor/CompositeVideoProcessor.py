import os
import os.path
import cv2
import numpy as np
from tqdm import tqdm
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from skimage.transform import resize
from sunback.processor.Processor import Processor


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

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.final_output_path = None
        self.frame_shape = None
        self.good_paths = {171: [], 193: [], 211: []}
        self.skipped = 0

    def do_work(self):
        """Main method to execute the composite video generation process."""
        if self.process_done:
            print("Composite video generation already completed. Skipping process.")
            return

        wavelengths = [171, 193, 211]  # List of wavelengths to process
        self.collect_fits_paths(wavelengths)  # Collect the FITS paths to process
        video_writer = self.init_writer()
        if video_writer:
            self.run_composite_video_writer(video_writer)
            self.process_done = True  # Mark the process as done

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
                self.good_paths[wavelength] = fits_files

        # Ensure all three wavelengths have the same number of FITS files
        num_frames = min(
            len(self.good_paths[171]),
            len(self.good_paths[193]),
            len(self.good_paths[211]),
        )
        for wavelength in wavelengths:
            self.good_paths[wavelength] = self.good_paths[wavelength][:num_frames]

    def init_writer(self):
        """Initialize the video writer."""
        self.build_output_path()

        if not self.frame_shape:
            if self.good_paths[171]:
                with fits.open(self.good_paths[171][0]) as hdul:
                    data = hdul[-1].data  # Use the last HDU
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
        if not os.path.exists(thepath):
            os.makedirs(os.path.dirname(thepath), exist_ok=True)
        os.makedirs(self.rainbow_path, exist_ok=True)
        return self.final_output_path

    def run_composite_video_writer(self, video_writer):
        """Generate the composite video file using the last HDU of FITS files."""
        last_valid_data = {
            171: None,
            193: None,
            211: None,
        }  # Store last valid frame data for each wavelength

        for i in tqdm(
            range(len(self.good_paths[171])), desc=self.progress_text, unit="frames"
        ):
            try:
                # Load data for the current frame from each wavelength
                data_171 = (
                    self.load_fits_data(self.good_paths[171][i])
                    if i < len(self.good_paths[171])
                    else None
                )
                data_193 = (
                    self.load_fits_data(self.good_paths[193][i])
                    if i < len(self.good_paths[193])
                    else None
                )
                data_211 = (
                    self.load_fits_data(self.good_paths[211][i])
                    if i < len(self.good_paths[211])
                    else None
                )

                # Check if any data is missing and handle gracefully
                if data_171 is None:
                    data_171 = (
                        last_valid_data[171]
                        if last_valid_data[171] is not None
                        else np.ones(self.frame_shape) * 0.5
                    )
                else:
                    last_valid_data[171] = data_171

                if data_193 is None:
                    data_193 = (
                        last_valid_data[193]
                        if last_valid_data[193] is not None
                        else np.ones(self.frame_shape) * 0.5
                    )
                else:
                    last_valid_data[193] = data_193

                if data_211 is None:
                    data_211 = (
                        last_valid_data[211]
                        if last_valid_data[211] is not None
                        else np.ones(self.frame_shape) * 0.5
                    )
                else:
                    last_valid_data[211] = data_211

                # Normalize and resize the data to match target frame shape
                norm_171 = self.normalize_and_resize(data_171)
                norm_193 = self.normalize_and_resize(data_193)
                norm_211 = self.normalize_and_resize(data_211)

                # Create RGB composite image
                img_rgb = make_lupton_rgb(
                    norm_171,
                    norm_193,
                    norm_211,
                    Q=0,
                    stretch=1,
                    # filename=self.rainbow_path + f"/{i}.png",
                )
                img_8bit = (img_rgb).astype(np.uint8)  # Convert to 8-bit

                video_writer.write(img_8bit)
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                self.skipped += 1

        video_writer.release()
        print(
            f" ^    Successfully {self.finished_verb} with {len(self.good_paths[171]) - self.skipped} frames! ({self.skipped} skipped)"
        )

    def load_fits_data(self, file_path):
        """Load data from the last HDU of a FITS file."""
        with fits.open(file_path) as hdul:
            return hdul[-1].data

    def normalize_and_resize(self, data):
        """Normalize and resize data to match the frame shape."""
        target_shape = (self.frame_shape[1], self.frame_shape[0])
        data_resized = resize(data, target_shape, preserve_range=True)
        return data_resized
        return (data_resized - np.min(data_resized)) / (
            np.max(data_resized) - np.min(data_resized)
        )
