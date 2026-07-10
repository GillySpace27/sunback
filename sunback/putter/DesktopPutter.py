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
    handlers=[logging.StreamHandler(sys.stdout)]
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
            self._set_wallpaper_macos(str(local_path))

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
                logger.info("Executing command: %s", " ".join(command))
                subprocess.run(command, check=True)

                command_refresh = ["RUNDLL32.EXE", "user32.dll,UpdatePerUserSystemParameters"]
                logger.info("Executing command: %s", " ".join(command_refresh))
                subprocess.run(command_refresh, check=True)

                logger.info("Wallpaper updated successfully.")

            except subprocess.CalledProcessError as e:
                logger.error("Failed to update wallpaper on Windows: %s", e)
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

    def _set_wallpaper_macos(self, path):
        """Set the macOS desktop wallpaper.

        Uses the native NSWorkspace API (via pyobjc) instead of driving System
        Events with osascript. On macOS 14+ the osascript path needs Automation
        permission — which a background LaunchAgent (lingon) can't be prompted
        to grant — so it silently no-ops. NSWorkspace runs in-process in the
        user's GUI session and returns an explicit success/error, so we trust
        that instead of reading the wallpaper back (the macOS getter lags by a
        set and can't be used to verify). Falls back to osascript only if
        pyobjc isn't installed.
        """
        try:
            from AppKit import NSWorkspace, NSScreen
            from Foundation import NSURL
        except ImportError:
            # ponytail: last-resort fallback; NSWorkspace is the real path.
            cmd = f'tell application "System Events" to tell every desktop to set picture to "{path}"'
            subprocess.run(["osascript", "-e", cmd], check=True)
            return

        url = NSURL.fileURLWithPath_(path)
        ws = NSWorkspace.sharedWorkspace()
        screens = NSScreen.screens()
        if not screens:
            raise OSError("No screens available — is there a GUI login session?")
        for screen in screens:
            ok, err = ws.setDesktopImageURL_forScreen_options_error_(url, screen, {}, None)
            if not ok:
                raise OSError(f"NSWorkspace failed to set wallpaper: {err}")

    def super_flush(self):
        """Force flush for better output handling."""
        sys.stdout.flush()