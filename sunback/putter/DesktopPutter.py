import sys
from os.path import abspath, split
from platform import system
import os
from tqdm import tqdm
import subprocess

from sunback.putter.Putter import Putter
# Initialization
from time import time, sleep

last_time = time()
start_time = last_time
set_local_background = True
test = False
# from src.utils.file_util import load_imgs_paths


class DesktopPutter(Putter):
    description = "Use the local images to set the desktop background"
    filt_name = "DesktopPutter"

    def put(self, params=None):
        self.load(params)
        sys.stdout.flush()
        print("\r V Setting Desktop Background to...(ctrl-c to skip)", flush=True)
        self.super_flush()
        to_display = sorted([file for file in self.params.local_imgs_paths()])

        if not len(to_display):
            filenames = os.listdir(self.params.imgs_top_directory())
            to_display = sorted([self.params.imgs_top_directory() + os.path.sep + file for file in filenames])

        try:
            im_171 = [file for file in to_display if "171" in file][0]
            to_display.append(im_171)
        except IndexError:
            print("171A not found")
            pass
        self.ii = 0
        for png_path in to_display:
            self.ii += 1
            self.update_background(png_path)
            self.sleep_until_delay_elapsed()
        print(" ^ Loop Complete", flush=True)

    def update_background(self, local_path):
        """
        Update the System Background

        Parameters
        ----------
        local_path : str
         The local save location of the in_object
         :param local_path:
         :return:
        """
        local_path = abspath(local_path)
        self.png_name = local_path.split("\\")[-1]
        # self.params.current_wave(self.png_name)
        # print(local_path)
        assert isinstance(local_path, str)
        # print("Updating Background...", pointing_end='', flush=True)
        this_system = system()

        if this_system == "Darwin":
            osascript_command = f'tell application "System Events" to set picture of every desktop to "{local_path}"'
            os.system(f"osascript -e '{osascript_command}'")

        elif this_system == "Windows":
            # Command to change wallpaper on Windows
            command = f'REG ADD "HKCU\Control Panel\Desktop" /V Wallpaper /T REG_SZ /F /D {local_path}'
            subprocess.run(command, shell=True)
            subprocess.run('RUNDLL32.EXE user32.dll,UpdatePerUserSystemParameters', shell=True)

        elif this_system == "Linux":
            # Command to change wallpaper on Linux (GNOME)
            command = f"gsettings set org.gnome.desktop.background picture-uri file://{local_path}"
            subprocess.run(command, shell=True)
        else:
            raise OSError("Operating System Not Supported")



        # try:
        #     if this_system == "Windows":
        #         import ctypes
        #         SPI_SETDESKWALLPAPER = 0x14  # which command (20)
        #         SPIF_UPDATEINIFILE = 0x2  # forces instant update
        #         ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, local_path, SPIF_UPDATEINIFILE)
        #         # for ii in np.arange(100):
        #         #     ctypes.windll.user32.SystemParametersInfoW(19, 0, 'Fit', SPIF_UPDATEINIFILE)
        #     elif this_system == "Darwin":
        #         # print(image_path)
        #         osascript_command = (
        #             f'tell application "System Events" to set picture of every desktop to "{local_path}"'
        #         )

        #         # osascript_command = (
        #         #     f'tell application "System Events" to '
        #         #     f'do shell script "sqlite3 ~/Library/Application\\\\ Support/Dock/desktoppicture.db '
        #         #     f'\\\"UPDATE data SET value = \'{local_path}\'\\\"; killall Dock"'
        #         #     f' with administrator privileges'
        #         # )

        #         os.system(f"osascript -e '{osascript_command}'")

        #     elif this_system == "Linux":
        #         os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-options 'scaled'")
        #         os.system("/usr/bin/gsettings set org.gnome.desktop.background primary-color 'black'")
        #         os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-uri {}".format(local_path))
        #     else:
        #         raise OSError("Operating System Not Supported")
        #     # print("Success")
        # except Exception as e:
        #     print("Failed")
        #     raise e
        #
        # if self.params.is_debug():
        #     self.plot_full_normalization()

        return 0
