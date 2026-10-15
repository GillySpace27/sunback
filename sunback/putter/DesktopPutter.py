import sys
import os
import platform
from pathlib import Path
from time import time, sleep
import subprocess
import logging
from sunback.putter.Putter import Putter  # Assuming this is a custom import
from sunback.client import wallpaper

# 2026-10-02 (SB-11): logging is configured by the entry point (run_client in
# sunback/run/run_client_background.py), not at import. Original block:
# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
logger = logging.getLogger(__name__)

# Initialization
last_time = time()
start_time = last_time
set_local_background = True
test = False


class DesktopPutter(Putter):
    description = "Use the local images to set the desktop background"
    filt_name = "DesktopPutter"

    def put(self, params=None):
        """
        Loops through the available images and sets them as the desktop background sequentially.
        """
        self.load(params)
        logger.info("Starting to set desktop backgrounds...")
        self.super_flush()

        # Get list of images to display
        to_display = sorted(self.params.local_imgs_paths())

        if not to_display:
            filenames = os.listdir(self.params.imgs_top_directory())
            to_display = sorted([os.path.join(self.params.imgs_top_directory(), file) for file in filenames])

        # Ensure "171A" image is included, if available
        try:
            im_171 = next(file for file in to_display if "171" in file)
            to_display.append(im_171)
        except StopIteration:
            logger.warning("'171A' image not found.")

        # Sequentially update desktop background
        self.ii = 0
        for png_path in to_display:
            try:
                self.ii += 1
                self.png_name = png_path
                self.update_background(png_path)
                self.sleep_until_delay_elapsed()
            except Exception as e:
                logger.error(f"Error updating background to {png_path}: {e}")

        logger.info("Desktop background update loop complete.")

    def update_background(self, local_path):
        """
        Update the system's desktop background.

        Parameters
        ----------
        local_path : str
            The local path to the image file to set as the background.

        Raises
        ------
        OSError
            If the operating system or desktop environment is unsupported.
        FileNotFoundError
            If the specified file does not exist.
        """
        # Ensure the path is absolute and valid
        local_path = Path(local_path).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"File not found: {local_path}")

        # SB-21, 2026-10-15: the macOS, Windows and Linux branches moved, unchanged,
        # to sunback/client/wallpaper.py, which the new client shares.
        wallpaper.set_wallpaper(str(local_path))

        logger.debug(f"Wallpaper updated successfully to {local_path}")

    def _set_wallpaper_macos(self, path):
        """Kept for callers of the old method; the code is wallpaper.set_wallpaper_macos (SB-21)."""
        wallpaper.set_wallpaper_macos(path)

    def super_flush(self):
        """Force flush for better output handling."""
        sys.stdout.flush()