from os.path import abspath, split
from platform import system

from tqdm import tqdm

from putter.Putter import Putter
# Initialization
from time import time, sleep

last_time = time()
start_time = last_time
set_local_background = True
from utils.file_util import load_img_paths

class DesktopPutter(Putter):
    def __init__(self, params):
        self.params = params
        self.delay = self.params.delay_seconds()
        self.params.stop_after_one(False)

    def load(self):
        load_img_paths(self.params)
    
    def put(self):
        self.load()
        print("  *Setting Desktop Background to...", flush=True)
        sleep(0.1)
        for png_path in self.params.local_img_paths():
            self.update_background(png_path)
            self.sleep_until_delay_elapsed()
        # print("Loop Complete", flush=True)
    
    def sleep_until_delay_elapsed(self):
        """ Make sure that the loop takes the right amount of time """
        delay = self.params.delay_seconds()
        for ii in tqdm((range(int(delay))),
                       desc="    {}, Waiting for {:0.0f} seconds".format(self.png_name, delay)):
            sleep(1)
    
    def update_background(self, local_path):
        """
        Update the System Background
        
        Parameters
        ----------
        local_path : str
         The local save location of the image
         :param local_path:
         :return:
        """
        local_path = abspath(local_path)
        self.png_name = local_path[-8:]
        # print(local_path)
        assert isinstance(local_path, str)
        # print("Updating Background...", end='', flush=True)
        this_system = system()
        
        try:
            if this_system == "Windows":
                import ctypes
                SPI_SETDESKWALLPAPER = 0x14  # which command (20)
                SPIF_UPDATEINIFILE = 0x2  # forces instant update
                ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, local_path, SPIF_UPDATEINIFILE)
                # for ii in np.arange(100):
                #     ctypes.windll.user32.SystemParametersInfoW(19, 0, 'Fit', SPIF_UPDATEINIFILE)
            elif this_system == "Darwin":
                # from appscript import app, mactypes
                # try:
                #     app('Finder').desktop_picture.set(mactypes.File(local_path))
                # except Exception as e:
                #     if test:
                #         pass
                #     else:
                #         raise e
                print("Screw you, Macintosh, this don't work here")
                pass
            elif this_system == "Linux":
                import os
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-options 'scaled'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background primary-color 'black'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-uri {}".format(local_path))
            else:
                raise OSError("Operating System Not Supported")
            # print("Success")
        except Exception as e:
            print("Failed")
            raise e
        #
        # if self.params.is_debug():
        #     self.plot_stats()
        
        return 0
