import sys
import os
import platform
from pathlib import Path
from time import time, sleep
import subprocess
import logging
from sunback.putter.Putter import Putter  # Assuming this is a custom import

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
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

        # Detect the operating system
        this_system = platform.system()

        # Platform-specific logic
        if this_system == "Darwin":  # macOS
            try:
                osascript_command = f'tell application "System Events" to set picture of every desktop to "{local_path}"'
                subprocess.run(["osascript", "-e", osascript_command], check=True)
            except subprocess.CalledProcessError as e:
                raise OSError(f"Failed to update wallpaper on macOS: {e}")

        elif this_system == "Windows":  # Windows
            try:
                # Update registry and refresh the wallpaper
                command = [
                    "REG",
                    "ADD",
                    r"HKCU\Control Panel\Desktop",
                    "/V",
                    "Wallpaper",
                    "/T",
                    "REG_SZ",
                    "/F",
                    "/D",
                    str(local_path),
                ]
                subprocess.run(command, shell=True, check=True)
                subprocess.run(["RUNDLL32.EXE", "user32.dll,UpdatePerUserSystemParameters"], shell=True, check=True)
            except subprocess.CalledProcessError as e:
                raise OSError(f"Failed to update wallpaper on Windows: {e}")

        elif this_system == "Linux":  # Linux
            try:
                desktop_env = os.getenv("XDG_CURRENT_DESKTOP", "").lower()
                if "gnome" in desktop_env:
                    subprocess.run(
                        ["gsettings", "set", "org.gnome.desktop.background", "picture-uri", f"file://{local_path}"],
                        check=True,
                    )
                elif "kde" in desktop_env:
                    raise NotImplementedError("KDE wallpaper update is not implemented.")
                elif "xfce" in desktop_env:
                    raise NotImplementedError("XFCE wallpaper update is not implemented.")
                else:
                    raise OSError(f"Unsupported desktop environment: {desktop_env}")
            except subprocess.CalledProcessError as e:
                raise OSError(f"Failed to update wallpaper on Linux: {e}")
        else:
            raise OSError(f"Unsupported operating system: {this_system}")

        logger.debug(f"Wallpaper updated successfully to {local_path}")

    def super_flush(self):
        """Force flush for better output handling."""
        sys.stdout.flush()