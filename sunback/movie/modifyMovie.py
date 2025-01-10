"""
sunback.py
A program that downloads the most current images of the sun from the SDO satellite,
then sets each of the images to the desktop background in series.

Handles the primary functions
"""

# Imports
from time import localtime, timezone, strftime, sleep, time, struct_time
from urllib.request import urlretrieve
from os import getcwd, makedirs, rename, remove, listdir, startfile
from os.path import normpath, abspath, join, dirname, exists
from calendar import timegm
import astropy.units as u

start = time()
from sunpy.net import Fido, attrs as a
# import sunpy.map
from sunpy.io import read_file_header, write_file
from moviepy.editor import AudioFileClip, VideoFileClip
import cv2
from pippi import tune
from functools import partial
from threading import Thread, Barrier
from copy import copy
bbb = Barrier(3, timeout=100)
from astropy.nddata import block_reduce
from hello_world.sunback_filt_movie import Modify


print("Import took {:0.2f} seconds".format(time() - start))
from parfive import Downloader
# import numba
# from numba import jit
from tqdm import tqdm
from warnings import warn
from platform import system
import sys
import numpy as np
import matplotlib as mpl

try:
    mpl.use('qt5agg')
except:
    pass
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
plt.ioff()

from playsound import playsound as ps
from pippi.oscs import Osc
from pippi import dsp, fx

this_system = system()

if this_system == "Windows":
    # Windows Imports
    import sunpy.visualization.colormaps

elif this_system == "Linux":
    # Linux Imports
    import sunpy.visualization.colormaps

elif this_system == "Darwin":
    # Mac Imports
    pass

else:
    raise OSError("Operating System Not Supported")

# Main

debugg = True


class SunbackMovie:
    """
    The Primary Class that Does Everything

    Parameters
    ----------
    parameters : Parameters (optional)
        a class specifying run options
    """


    def run(self, delay=20, mode='all', debug=False):
        p = Parameters()
        p.mode(mode)
        p.set_delay_seconds(delay)
        p.do_mirror(False)
        # p.do_171(True)

        if debug:
            p.is_debug(True)
            p.set_delay_seconds(10)
            p.do_HMI(False)

        # p.time_period(period=['2019/12/21 04:20', '2019/12/21 04:40'])
        p.resolution(1024)
        p.range(days=3)#0.060)
        p.download_images(True)
        p.cadence(30)
        p.frames_per_second(20)
        p.bpm(150)
        # p.download_images(False)
        # p.overwrite_pngs(False)
        p.sonify_limit(False)
        p.remove_old_images(False)
        p.make_compressed(True)
        p.sonify_images(True, True)
        # p.sonify_images(False, False)
        # p._stop_after_one = True
        # p.do_171(True)
        # p.do_304(True)

        # Sunback(p).start()
        SunbackMovie(p).start()


    def where():
        """Prints the location that the images are stored in."""
        p = Parameters()
        print(p.discover_best_default_directory())


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    where()
    run(20, 'y', debug=debugg)












# cdef int SINE = 0
# cdef int SINEIN = 17
# cdef int SINEOUT = 18
# cdef int COS = 1
# cdef int TRI = 2
# cdef int SAW = 3
# cdef int RSAW = 4
# cdef int HANN = 5
# cdef int HANNIN = 21
# cdef int HANNOUT = 22
# cdef int HAMM = 6
# cdef int BLACK = 7
# cdef int BLACKMAN = 7
# cdef int BART = 8
# cdef int BARTLETT = 8
# cdef int KAISER = 9
# cdef int SQUARE = 10
# cdef int RND = 11
# cdef int LINE = SAW
# cdef int PHASOR = SAW
# cdef int SINC = 23
# cdef int GAUSS = 24
# cdef int GAUSSIN = 25
# cdef int GAUSSOUT = 26
# cdef int PLUCKIN = 27
# cdef int PLUCKOUT = 28