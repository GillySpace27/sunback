
import numpy as np
# import sunpy.map
import os
import matplotlib.pyplot as plt
from time import strftime
from datetime import timedelta
from scipy.signal import savgol_filter

import datetime

"""
sunback.py
A program that downloads the most current images of the sun from the SDO satellite,
then sets each of the images to the desktop background in series.

Handles the primary functions
"""

# Imports
from time import localtime, timezone, strftime, sleep, time, struct_time
# from urllib.request import urlretrieve
from os import getcwd, makedirs, rename, remove, listdir
from os.path import normpath, abspath, join, dirname, exists
from calendar import timegm
import astropy.units as u

start = time()
from sunpy.net import Fido, attrs as a
# import sunpy.map
from sunpy.io import read_file_header, write_file
# from moviepy.editor import AudioFileClip, VideoFileClip
# import cv2
# from pippi import tune
from functools import partial
from threading import Thread, Barrier
from copy import copy
bbb = Barrier(3, timeout=100)

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

# from playsound import playsound as ps
# from pippi.oscs import Osc
# from pippi import dsp, fx

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

debugg = False




class ModifyMovie:
    """
    The Primary Class that Does Everything

    Parameters
    ----------
    parameters : Parameters (optional)
        a class specifying run options
    """

    def __init__(self):
        """Initialize a new parameter object or use the provided one"""
        self.fits_analysis_done = False
        self.indexNow = 0
        self.last_time = 0
        self.this_time = 1
        self.new_images = False
        self.fido_result = None
        self.fido_num = None
        self.renew_mask = True
        self.mask_num = [1, 2]
        self.wavelengths = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700']
        self.waveNum = len(self.wavelengths)
        self.this_name = None
        self.resume = True
        self.sonify_complete = False
        self.proper_bin=None

        if not self.params.is_debug():
            sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')

        self.start()

    # # Main Command Structure  ##############################################
    def start(self):
        """Select whether to run or to debug"""
        self.print_header()

        if self.params.is_debug():
            self.debug_mode()
        else:
            self.run_mode()

    def debug_mode(self):
        """Run the program in a way that will break"""
        while True:
            self.execute()

    def run_mode(self):
        """Run the program in a way that won't break"""

        fail_count = 0
        fail_max = 10

        while True:
            try:
                self.execute()
            except (KeyboardInterrupt, SystemExit):
                print("\n\nOk, I'll Stop. Doot!\n")
                break
            except Exception as error:
                fail_count += 1
                if fail_count < fail_max:
                    print("I failed, but I'm ignoring it. Count: {}/{}\n\n".format(fail_count, fail_max))
                    self.need = True
                    continue
                else:
                    print("Too Many Failures, I Quit!")
                    sys.exit(1)

    def print_header(self):
        print("\nSunback: SDO Video Maker \nWritten by Chris R. Gilly")
        print("Check out my website: http://gilly.space\n")

    def execute(self):

        for ii in np.arange(self.waveNum):
            self.main_loop(ii)

    # # Main Functions  ######################################################

    def main_loop(self, ii):
        """The Main Loop"""

        # Initialize Everything
        if self.init_or_skip(ii):
            return

        # Download the new fits data
        self.space_to_fits()

        # Remove the old fits files
        self.remove_all_old_files()

        # Analyze the Dataset as a whole
        if False:
            self.fits_analyze_whole_set()

        # Generate the png images
        self.fits_to_pngs()

        # Generate the Movie
        self.pngs_to_movie()

        # Generate the Audio
        self.fits_to_audio()

        # Add Sound to the Movie
        self.movie_to_audio_movie()

        self.soni.thread_lock()
        print("Wavelength Complete: {}, took {:0.2f} minutes\n\n".format(self.this_name, (time() - self.beginTime) / 60))

        if self.params.stop_after_one():
            sys.exit()

    def init_or_skip(self, ii):
        """Initializes the instance variables to this index if the flags allow it, else skips"""
        if self.params.do_171():
            ii = 2

        if self.params.do_304():
            ii = 5

        self.this_name = self.wavelengths[ii]

        if self.skip():
            return True

        self.beginTime = time()
        self.save_path = join(self.params.local_directory, self.this_name)
        makedirs(self.save_path, exist_ok=True)
        self.local_path = self.params.local_directory
        self.image_folder = self.save_path
        self.movie_folder = self.save_path
        self.video_name_stem = join(self.movie_folder, '{}_movie{}'.format(self.this_name, '{}'))

        name = "{}_{}".format(self.this_name, "max")
        self.soni = Sonifier(self.params, self.save_path, name, self.video_name_stem, frames_per_second=self.params.frames_per_second())

        print("\nMovie: {}".format(self.this_name))
        return False

    def space_to_fits(self):
        """ Find the Most Recent Images """
        # Define Time Range
        self.get_time_range()

        if self.params.download_images():
            print(">Acquiring Science Images from {} to {}...".format(self.earlyString, self.nowString), flush=True)

            # Search for records from the internet
            self.fido_result = Fido.search(a.Time(self.early, self.now), a.Instrument('aia'), a.Wavelength(int(self.this_name) * u.angstrom),
                                           a.vso.Sample(self.params.cadence()), a.Level(1.5))

            # See what we already have
            files = listdir(self.save_path)
            already_downloaded = []
            for filename in files:
                if filename.endswith(".fits") and "norm" not in filename:
                    already_downloaded.append(int(self.time_from_filename(filename)[0]))

            # Define what we still need
            to_get = np.empty(self.fido_result.file_num)
            to_get.fill(True)
            for ii, file_to_get in enumerate(self.fido_result.get_response(0)):
                tm = int(file_to_get.time.start)
                if tm in already_downloaded:
                    to_get[ii] = False
            getNum =int(np.sum(to_get)) - 1
            need = np.nonzero(to_get)[0]

            # Analyze Results
            self.import_fido_information()
            print("  Search Found {} Images {}...".format(self.fido_num, self.extra_string), flush=True)

            if getNum > 1:
                self.new_images = True
                print("    Downloading {} New:".format(getNum), flush=True)

                #Make groups
                start = need[0]
                end=start
                box2=[]
                box1 = []

                for ii in np.arange(len(need)-1):
                    end = need[ii]+1
                    if need[ii] == need[ii+1] - 1:
                        continue
                    box2.append(self.fido_result[0, start:end])
                    box1.append((start,end))
                    start = need[ii+1]
                if end - start > 1:
                    box2.append(self.fido_result[0, start:end])
                    box1.append((start,end))

                for st in box2:
                    Fido.fetch(st, path=self.save_path, downloader=Downloader(progress=True, file_progress=False), overwrite=True)
                # print("Short took {:0.3f} seconds.".format(time()-startT))
            else:
                self.new_images = False
                print("   Success: All Images Already Downloaded", flush=True)


    def get_time_range(self):
        """Define Time Range, on the hour"""
        # Get the Start Time
        current_time = time() + timezone
        start_list = list(localtime(current_time - (self.params.range() + 2 / 24) * 60 * 60 * 24))

        # Truncate the minutes and seconds
        # start_list[2] -= 0 # Days

        start_list[4] = 0  # Minutes
        start_list[5] = 0  # Seconds
        start_struct = struct_time(start_list)

        # Make Output Products
        self.early = strftime('%Y/%m/%d %H:%M', start_struct)
        self.earlyLong = int(strftime('%Y%m%d%H%M%S', start_struct))
        self.earlyString = self.parse_time_string_to_local(str(self.earlyLong), 2)

        # Get the Current Time
        now_list = list(localtime(current_time - 2 * 60 * 60))

        # Truncate the minutes and seconds
        now_list[4] = 0  # Minutes
        now_list[5] = 0  # Seconds
        now_struct = struct_time(now_list)

        # Make Output Products
        self.now = strftime('%Y/%m/%d %H:%M', now_struct)
        self.nowLong = int(strftime('%Y%m%d%H%M%S', now_struct))
        self.nowString = self.parse_time_string_to_local(str(self.nowLong), 2)

    def fits_analyze_whole_set(self):
        """Check several fits files to determine fit curves"""
        print("Analyzing Dataset...", flush=True, end='')
        max_analyze = 5
        minBox = []
        maxBox = []

        self.save_path = join(self.params.local_directory, self.this_name)

        ii = 0
        for filename in listdir(self.save_path):
            if filename.endswith(".fits"):
                ii += 1
                image_path = join(self.save_path, filename)
                fname = filename[3:18]
                fname = fname.replace("_", "")
                time_string = self.parse_time_string_to_local(fname, 2)

                # Load the File
                originalData, single_image_data = self.load_fits_series((self.this_name, image_path, time_string))

                self.radial_analyze(originalData, False)
                minBox.append(self.fakeMin)
                maxBox.append(self.fakeMax)

                print('.', end='')
                if ii >= max_analyze:
                    self.fakeMin = np.mean(np.asarray(minBox), axis=0)
                    self.fakeMax = np.mean(np.asarray(maxBox), axis=0)
                    self.fits_analysis_done = True
                    print("Done!")
                    break

    def fits_to_pngs(self):
        """Re-save all the Fits images into pngs and normed fits files"""
        self.apply_func_to_directory(self.do_image_work, doAll=False, desc=">Processing Images", unit="images")

    def fits_to_audio(self):
        """Analyzes the fits files to sonify them"""
        if self.params.sonify_images() and not self.sonify_complete:
            self.apply_func_to_directory(self.do_sonifying_work, doAll=True, desc=">Sonifying Images", unit="images", limit=self.params.sonify_limit())
        self.soni.generate_track(self.soni.wav_path)
        # self.soni.play()

    def pngs_to_movie(self):
        """Combines all png files into an avi movie"""
        try:
            videoclip_full = VideoFileClip(self.video_name_stem.format("_raw.avi"))
            invalid_movie=False
        except:
            invalid_movie=True

        if self.new_images or invalid_movie:
            # logger = open(join(self.params.local_directory, 'log.txt'), 'w+')
            try:
                images = [img for img in listdir(self.image_folder) if img.endswith(".png") and self.check_valid_png(img)]
                if len(images) > 0:
                    frame = cv2.imread(join(self.image_folder, images[0]))
                    height, width, layers = frame.shape
                    video_avi = cv2.VideoWriter(self.video_name_stem.format("_raw.avi"), 0, self.params.frames_per_second(), (width, height))

                    for image in tqdm(images, desc=">Writing Movie", unit="frame"):
                        # Delete it if it is too old
                        im = cv2.imread(join(self.image_folder, image))
                        # import pdb; pdb.set_trace()
                        video_avi.write(im)


                    cv2.destroyAllWindows()
                    video_avi.release()

                else:
                    print("No png Images Found")
            except FileNotFoundError:
                print("Images Not Found")

    def movie_to_audio_movie(self):
        """Multiplexes the generated wav and avi files into a single movie"""
        if self.params.allow_muxing() and (self.new_images or self.params.sonify_images()):
            print(">Muxing Main Movie...")
            videoclip_full = VideoFileClip(self.video_name_stem.format("_raw.avi"))
            videoclip_full_muxed = videoclip_full.set_audio(AudioFileClip(self.soni.wav_path))
            from proglog import TqdmProgressBarLogger

            hq_sonFunc = partial(videoclip_full_muxed.write_videofile, self.video_name_stem.format("_HQ.mp4"), codec='libx264', bitrate='400M',
                                 logger=TqdmProgressBarLogger(print_messages=True))
            t1 = Thread(target=hq_sonFunc)
            t1.start()
            t1.join()


            if self.params.make_compressed():
                clip = videoclip_full_muxed
                lq_sonFunc = partial(clip.write_videofile, self.video_name_stem.format("_LQ.mp4"), codec='libx264', bitrate='50M')
                wq_sonFunc = partial(clip.write_videofile, self.video_name_stem.format(".webm"), codec='libvpx')
                t2 = Thread(target=lq_sonFunc)
                t3 = Thread(target=wq_sonFunc)
                t2.start()
                t2.join()
                t3.start()
                t3.join()



            print("  Successfully Muxed")
            # # Play the Movie
            # startfile(self.video_name_stem.format("_HQ.mp4"))

    # # Support Functions  ###################################################

    def skip(self):

        # Skip Logic
        # if '4500' in self.this_name:
        #     return
        # if self.this_name in ['0094'] and self.params.is_first_run:
        #     # print("Skip for Now\n")
        #     return 1
        # #
        # # if self.params.is_debug():
        # if self.this_name in ['0193']:
        #     sys.exit()
        # #
        # # # if int(self.this_name) < 1000:
        # # #     return

        # if '304' not in self.this_name:
        #     return 1
        return 0

    def import_fido_information(self):
        self.fido_num = self.fido_result.file_num

        time_start = self.fido_result.get_response(0)[0].time.start
        time_end = self.fido_result.get_response(0)[-1].time.start

        self.startTime = self.parse_time_string_to_local(str(int(time_start)), 2)
        self.endTime = self.parse_time_string_to_local(str(int(time_end)), 2)

        self.name = self.fido_result.get_response(0)[0].wave.wavemin
        while len(self.name) < 4:
            self.name = '0' + self.name

        if self.params.is_debug():
            self.extra_string = "from {} to {}".format(self.startTime, self.endTime)
        else:
            self.extra_string = ''

    def define_single_image(self, filename):
        time_code, time_string = self.time_from_filename(filename)
        image_path = join(self.save_path, filename)
        single_image_data = (self.this_name, image_path, time_string, time_code, filename)
        return single_image_data

    def remove_all_old_files(self):
        files = listdir(self.save_path)
        file_idx = 0
        for filename in files:
            if filename.endswith(".fits") and "norm" not in filename:
                if self.remove_old_files(self.define_single_image(filename)):
                    file_idx += 1
                    continue
        if file_idx > 0:
            if self.params.remove_old_images():
                print("Deleted {} old images".format(file_idx))
            # else:
            #     print("Excluding {} old images".format(file_idx))

    def remove_old_files(self, single_image_data):
        filename = single_image_data[4]
        thisTime = int(self.time_from_filename(filename)[0])
        if thisTime < self.earlyLong:
            if self.params.remove_old_images():
                self.deleteFiles(filename)
            return 1
        return 0

    def this_frame_is_bad(self, image_array, single_image_data):
        filename = single_image_data[4]
        total_counts = np.nansum(image_array)
        if total_counts < 0:
            self.deleteFiles(filename)
            return 1
        return 0

    def apply_func_to_directory(self, func, doAll=False, desc="Work Done", unit="it", limit=False):
        work_list = []
        files = listdir(self.save_path)
        file_idx = -1
        for filename in files:
            if filename.endswith(".fits") and "norm" not in filename:
                # Define the image
                single_image_data = self.define_single_image(filename)
                pngPath = join(self.save_path, filename[:-4] + 'png')

                # Delete it if it is too old
                if self.remove_old_files(single_image_data):
                    continue

                file_idx += 1
                if doAll or not exists(pngPath) or self.params.overwrite_pngs():
                    if not limit or self.soni.frame_on_any_beat(file_idx):
                        work_list.append([single_image_data, file_idx])
                    else:
                        work_list.append(None)

        self.nRem = len(work_list)

        if self.nRem > 0:
            with tqdm(total=self.nRem, desc=desc, unit=unit) as pbar:
                for image in work_list:
                    if image:
                        func(image)
                    pbar.update()
            # # pbar.close()
            # from pymp import Parallel
            # with tqdm(total=self.nRem, desc=desc, unit=unit) as pbar:
            #     with Parallel(self.nRem) as p:
            #         for i in p.range(self.nRem):
            #             image = work_list[i]
            #             if image:
            #                 func(image)
            #             pbar.update()
            #     # pbar.close()
            # from joblib import Parallel, delayed

            # results = Parallel(n_jobs=-1, verbose=verbosity_level, backend="threading")(
            #     map(delayed(myfun), arg_instances))

            # import threading as mp
            # from multiprocessing.pool import ThreadPool
            # pool = ThreadPool()
            # pool.map(func, work_list)

    def do_image_work(self, single_image_data_ID):
        single_image_data, file_idx = single_image_data_ID
        # Load the File, destroying it if it fails
        fail, raw_image = self.load_fits_series(single_image_data)
        if fail:
            return 1

        # # Remove bad frames
        if self.this_frame_is_bad(raw_image, single_image_data):
            return 1

        # Modify the data
        processed_image_stats = self.image_modify(raw_image)

        # Sonify the data
        if False: #self.params.sonify_images():
            self.do_sonifying_work(single_image_data_ID, processed_image_stats, raw_image)
            self.sonify_complete=True if not self.params.download_images() else False

        # Plot and save the Data
        self.plot_and_save(processed_image_stats, single_image_data, raw_image)
        self.new_images = True
        return 0

    def do_sonifying_work(self, single_image_data_ID, proc_image_stats=None, raw_image=None):
        single_image_data, file_idx = single_image_data_ID

        if raw_image is None:
            # Load the File, destroying it if it fails

            fail1, raw_image = self.load_fits_series(single_image_data)

            # # Remove bad frames
            if fail1 or self.this_frame_is_bad(raw_image, single_image_data):
                return 1

        if proc_image_stats is None:
            single_image_data_proc = list(single_image_data)
            # print(single_image_data_proc)
            # import pdb; pdb.set_trace()
            single_image_data_proc[1] = single_image_data_proc[1][:-5] + "_norm.fits"
            fail2, proc_image_stats = self.load_fits_series(single_image_data_proc)

        # Sonify the data
        self.soni.sonify_frame(proc_image_stats, raw_image, file_idx)

        return 0

    def load_fits_series(self, image_data):
        # Load the Fits File from disk
        full_name, save_path, time_string, time_code, filename = image_data
        try:
            # Parse Inputs
            my_map = sunpy.map.Map(save_path)
        except (TypeError, OSError) as e:
            remove(save_path)
            return 1, 1

        data = my_map.data
        return 0, data

    def deleteFiles(self, filename):
        fitsPath = join(self.save_path, filename[:-5] + '.fits')
        pngPath = join(self.save_path, filename[:-5] + '.png')
        fitsPath2 = join(self.save_path, filename[:-5] + '_norm.fits')

        try:
            remove(fitsPath)
        except:
            pass
        try:
            remove(fitsPath2)
        except:
            pass
        try:
            remove(pngPath)
        except:
            pass

    def time_from_filename(self, filename):
        fname = filename[3:18]
        time_code = fname.replace("_", "")
        time_string = self.parse_time_string_to_local(time_code, 2)
        return time_code, time_string

    def check_valid_png(self, img):
        image_is_new=(int(self.time_from_filename(img)[0])) < self.earlyLong
        return not image_is_new

    def image_modify(self, data):
        data = data + 0
        self.radial_analyze(data, False)
        data = self.absqrt(data)
        data = self.coronagraph(data)
        data = self.vignette(data)
        data = self.append_stats(data)
        return data

    def append_stats(self, data):
        from scipy.signal import savgol_filter
        rank = 1
        window1 = 31
        window2 = 41
        mode = 'mirror'
        btma = self.binBtm[::self.extra_rez]
        mina = self.binMin[::self.extra_rez]
        mida = self.binMid[::self.extra_rez]
        maxa = self.binMax[::self.extra_rez]
        topa = self.binTop[::self.extra_rez]

        btma = savgol_filter(btma, window1, rank, mode=mode)
        mina = savgol_filter(mina, window1, rank, mode=mode)
        mida = savgol_filter(mida, window1, rank, mode=mode)
        maxa = savgol_filter(maxa, window1, rank, mode=mode)
        topa = savgol_filter(topa, window1, rank, mode=mode)

        btma = savgol_filter(btma, window2, rank, mode=mode)
        mina = savgol_filter(mina, window2, rank, mode=mode)
        mida = savgol_filter(mida, window2, rank, mode=mode)
        maxa = savgol_filter(maxa, window2, rank, mode=mode)
        topa = savgol_filter(topa, window2, rank, mode=mode)

        stacked = np.vstack(
            (data, btma, mina, mida, maxa, topa))
        return stacked

    def plot_and_save(self, data, image_data, original_data=None, ii=None):
        full_name, save_path, time_string, time_code, filename = image_data
        name = self.clean_name_string(full_name)

        for processed in [True]:

            if not self.params.is_debug():
                if not processed:
                    continue
            if not processed:
                if original_data is None:
                    continue

            # Save the Fits File
            header = read_file_header(save_path)[0]
            if "BLANK" in header.keys():
                del header["BLANK"]
            path = save_path[:-5] + '_norm.fits'
            write_file(path, np.asarray(data, dtype=np.float32), header, overwrite=True)

            data, _ = self.soni.remove_stats(data)

            # Create the Figure
            fig, ax = plt.subplots()
            self.blankAxis(ax)

            # inches = 10
            # fig.set_size_inches((inches, inches))
            #
            # pixels = data.shape[0]
            # dpi = pixels / inches
            # cocaine
            siX = 10
            siY = 10
            piX = 1080
            piY = 1080
            dpX = piX / siX
            dpY = piY / siY
            dpi = np.max((dpX, dpY))
            fig.set_size_inches((siX, siY))

            if 'hmi' in name.casefold():
                inst = ""
                plt.imshow(data, origin='upper', interpolation=None)
                # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                plt.tight_layout(pad=5.5)
                height = 1.05

            else:
                inst = '  AIA'
                plt.imshow(data if processed else self.normalize(original_data), cmap='sdoaia{}'.format(name),
                           origin='lower', interpolation=None,
                           vmin=self.vmin_plot, vmax=self.vmax_plot)
                plt.tight_layout(pad=0)
                height = 0.95

            # Annotate with Text
            buffer = '' if len(name) == 3 else '  '
            buffer2 = '    ' if len(name) == 2 else ''

            title = "{}    {} {}, {}{}".format(buffer2, inst, name, time_string, buffer)
            title2 = "{} {}, {}".format(inst, name, time_string)
            ax.annotate(title, (0.125, height + 0.02), xycoords='axes fraction', fontsize='large',
                        color='w', horizontalalignment='center')
            # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
            # the_time = strftime("%I:%M%p").lower()
            # if the_time[0] == '0':
            #     the_time = the_time[1:]
            # ax.annotate(the_time, (0.125, height), xycoords='axes fraction', fontsize='large',
            #             color='w', horizontalalignment='center')

            # Format the Plot and Save
            self.blankAxis(ax)
            middle = '' if processed else "_orig"
            new_path = save_path[:-5] + middle + ".png"

            if ii is not None and self.nRem > 0:
                remString = "{} / {} , {:0.1f}%".format(ii, self.nRem, 100 * ii / self.nRem)
            else:
                remString = ""

            try:
                plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi, compression=2, filter=None)
                # print("\tSaved {} Image {}, {} of {}   ".format('Processed' if processed else "Unprocessed", time_string, remString, self.this_name), end="\r")
            except PermissionError:
                new_path = save_path[:-5] + "_b.png"
                plt.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
                print("Success")
            except Exception as e:
                print("Failed...using Cached")
                if self.params.is_debug():
                    raise e
            plt.close(fig)

        return new_path

    def update_background(self, local_path, test=False):
        """
        Update the System Background

        Parameters
        ----------
        local_path : str
            The local save location of the image
        """
        print("Updating Background...", end='', flush=True)
        assert isinstance(local_path, str)
        local_path = normpath(local_path)

        this_system = platform.system()

        try:
            if this_system == "Windows":
                import ctypes
                ctypes.windll.user32.SystemParametersInfoW(20, 0, local_path, 0)
                # ctypes.windll.user32.SystemParametersInfoW(19, 0, 'Fill', 0)

            elif this_system == "Darwin":
                from appscript import app, mactypes
                try:
                    app('Finder').desktop_picture.set(mactypes.File(local_path))
                except Exception as e:
                    if test:
                        pass
                    else:
                        raise e

            elif this_system == "Linux":
                import os
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-options 'scaled'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background primary-color 'black'")
                os.system("/usr/bin/gsettings set org.gnome.desktop.background picture-uri {}".format(local_path))
            else:
                raise OSError("Operating System Not Supported")
            print("Success")
        except Exception as e:
            print("Failed")
            raise e

        if self.params.is_debug():
            self.plot_stats()

        return 0

    # Level 4

    @staticmethod
    def list_files1(directory, extension):
        from os import listdir
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    def get_paths(self, this_result):
        self.name = this_result.get_response(0)[0].wave.wavemin
        while len(self.name) < 4:
            self.name = '0' + self.name
        file_name = '{}_Now.fits'.format(self.name)
        save_path = join(self.params.local_directory, file_name)
        return self.name, save_path

    @staticmethod
    def parse_time_string_to_local(downloaded_files, which=0):
        if which == 0:
            time_string = downloaded_files[0][-25:-10]
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = int(time_string[9:11])
            minute = time_string[11:13]
        else:
            time_string = downloaded_files
            year = time_string[:4]
            month = time_string[4:6]
            day = time_string[6:8]
            hour_raw = time_string[8:10]
            minute = time_string[10:12]

        struct_time = (int(year), int(month), int(day), int(hour_raw), int(minute), 0, 0, 0, -1)

        new_time_string = strftime("%I:%M%p %m/%d/%Y", localtime(timegm(struct_time))).lower()
        if new_time_string[0] == '0':
            new_time_string = new_time_string[1:]

        # print(year, month, day, hour, minute)
        # new_time_string = "{}:{}{} {}/{}/{} ".format(hour, minute, suffix, month, day, year)

        return new_time_string

    @staticmethod
    def clean_name_string(full_name):
        # Make the name strings
        name = full_name + ''
        while name[0] == '0':
            name = name[1:]
        return name

    @staticmethod
    def blankAxis(ax):
        ax.patch.set_alpha(0)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='none', which='both',
                       top=False, bottom=False, left=False, right=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Data Manipulations

    @staticmethod
    def reject_outliers(data):
        # # Reject Outliers
        # a = data.flatten()
        # remove_num = 20
        # ind = argpartition(a, -remove_num)[-remove_num:]
        # a[ind] = nanmean(a)*4
        # data = a.reshape(data.shape)

        data[data > 10] = np.nan

        return data

    @staticmethod
    def absqrt(data):
        return np.sqrt(np.abs(data))

    @staticmethod
    def normalize(data):
        high = 98
        low = 15

        lowP = np.nanpercentile(data, low)
        highP = np.nanpercentile(data, high)
        return (data - lowP) / (highP - lowP)

    def vignette(self, data):

        mask = self.radius > (self.noise_radii)
        data[mask] = np.nan
        return data

    def vignette2(self, data):

        mask = np.isclose(self.radius, self.tRadius, atol=2)
        data[mask] = 1

        mask = np.isclose(self.radius, self.noise_radii, atol=2)
        data[mask] = 1
        return data

    def coronagraph(self, data):
        original = sys.stderr
        sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')

        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        dat_corona = (self.dat_flat - self.fakeMin[radius_bin]) / \
                     (self.fakeMax[radius_bin] - self.fakeMin[radius_bin])

        sys.stderr = original

        # Deal with too hot things
        self.vmax = 1
        self.vmax_plot = 1.9

        hot = dat_corona > self.vmax
        dat_corona[hot] = dat_corona[hot] ** (1 / 2)

        # Deal with too cold things
        self.vmin = 0.06
        self.vmin_plot = -0.03

        cold = dat_corona < self.vmin
        dat_corona[cold] = -((np.abs(dat_corona[cold] - self.vmin) + 1) ** (1 / 2) - 1) + self.vmin

        self.dat_coronagraph = dat_corona
        dat_corona_square = dat_corona.reshape(data.shape)

        if self.renew_mask or self.params.mode() == 'r':
            self.corona_mask = self.get_mask(data)
            self.renew_mask = False

        data = self.normalize(data)

        data[self.corona_mask] = dat_corona_square[self.corona_mask]

        # inds = np.argsort(self.rad_flat)
        # rad_sorted = self.rad_flat[inds]
        # dat_sort = dat_corona[inds]
        #
        # plt.figure()
        # # plt.yscale('log')
        # plt.scatter(rad_sorted[::30], dat_sort[::30], c='k')
        # plt.show()

        return data

    def get_mask(self, dat_out):

        corona_mask = np.full_like(dat_out, False, dtype=bool)
        rezz = corona_mask.shape[0]
        half = int(rezz / 2)

        mode = self.params.mode()

        if type(mode) in [float, int]:
            mask_num = mode
        elif 'y' in mode:
            mask_num = 1
        elif 'n' in mode:
            mask_num = 2
        else:
            if 'r' in mode:
                if len(mode) < 2:
                    mode += 'a'

            if 'a' in mode:
                top = 8
                btm = 1
            elif 'h' in mode:
                top = 6
                btm = 3
            elif 'd' in mode:
                top = 8
                btm = 7
            elif 'w' in mode:
                top = 2
                btm = 1
            else:
                print('Unrecognized Mode')
                top = 8
                btm = 1

            ii = 0
            while True:
                mask_num = np.random.randint(btm, top + 1)
                if mask_num not in self.mask_num:
                    self.mask_num.append(mask_num)
                    break
                ii += 1
                if ii > 10:
                    self.mask_num = []

        if mask_num == 1:
            corona_mask[:, :] = True

        if mask_num == 2:
            corona_mask[:, :] = False

        if mask_num == 3:
            corona_mask[half:, :] = True

        if mask_num == 4:
            corona_mask[:half, :] = True

        if mask_num == 5:
            corona_mask[:, half:] = True

        if mask_num == 6:
            corona_mask[:, :half] = True

        if mask_num == 7:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True

        if mask_num == 8:
            corona_mask[half:, half:] = True
            corona_mask[:half, :half] = True
            corona_mask = np.invert(corona_mask)

        return corona_mask

    # Basic Analysis

    def radial_analyze(self, data, plotStats=False):

        self.offset = np.abs(np.min(data))
        data += self.offset
        self.make_radius(data)
        self.sort_radially(data)

        stats = self.better_bin_stats(self.rad_sorted, self.dat_sorted, self.rez, self.offset)
        self.binBtm, self.binMin, self.binMax, self.binMid, self.binTop = stats

        if not self.fits_analysis_done:
            self.fit_curves()

        if plotStats:
            self.plot_stats()

    def make_radius(self, data):

        self.rez = data.shape[0]
        centerPt = self.rez / 2
        xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
        xc, yc = xx - centerPt, yy - centerPt

        self.extra_rez = 2

        self.sRadius = 400 * self.extra_rez
        self.tRadius = self.sRadius * 1.28
        self.radius = np.sqrt(xc * xc + yc * yc) * self.extra_rez
        self.rez *= self.extra_rez

    def sort_radially(self, data):
        # Create arrays sorted by radius
        self.rad_flat = self.radius.flatten()
        self.dat_flat = data.flatten()
        inds = np.argsort(self.rad_flat)
        self.rad_sorted = self.rad_flat[inds]
        self.dat_sorted = self.dat_flat[inds]



    @staticmethod
    # @numba.jit(nopython=True, parallel=True)
    def better_bin_stats(rad_sorted, dat_sorted, rez, offset):
        proper_bin = np.asarray(np.floor(rad_sorted), dtype=np.int32)
        binBtm = np.empty(rez);
        binBtm.fill(np.nan)
        binMin = np.empty(rez);
        binMin.fill(np.nan)
        binMax = np.empty(rez);
        binMax.fill(np.nan)
        binMid = np.empty(rez);
        binMid.fill(np.nan)
        binTop = np.empty(rez);
        binTop.fill(np.nan)

        bin_list = [np.float64(x) for x in range(0)]
        last = 0
        for ii in np.arange(len(proper_bin)):
            binInd = proper_bin[ii]
            if binInd != last:
                bin_array = np.asarray(bin_list)
                finite = bin_array[np.isfinite(bin_array)]
                data_in_bin = finite[np.nonzero(finite - offset)]
                if len(data_in_bin) > 0:
                    out = np.percentile(data_in_bin, [0.001, 2, 50, 95, 99.999])
                    binBtm[last], binMin[last], binMid[last], binMax[last], binTop[last] = out
                bin_list = []
            bin_list.append(dat_sorted[ii])
            last = binInd
        return binBtm, binMin, binMax, binMid, binTop

        # self.radBins = [[] for x in np.arange(self.rez)]
        # for ii, binI in enumerate(self.proper_bin):
        #     self.radBins[binI].append(self.dat_sorted[ii])
        #
        # # Find the statistics by bin
        # for bin_count, bin_list in enumerate(self.radBins):
        #     self.bin_the_slice(bin_count, bin_list)

        # i = 0
        # bin_count = 0
        # not_edge = self.proper_bin[:-1] == self.proper_bin[1:]
        # theRez = len(self.proper_bin)
        # myList = []
        # while i < theRez:
        #     if i < theRez - 1 and not_edge[i]:
        #         i += 1
        #         myList.append(self.dat_sorted[i])
        #         continue
        #     self.bin_the_slice(bin_count, np.asarray(myList))
        #     bin_count += 1
        #     i += 1
        # i = 0
        # i_prev = 0
        # bin_count = 0
        # not_edge = self.proper_bin[:-1] == self.proper_bin[1:]
        # theRez = len(self.proper_bin)
        # myList = []
        # while i < theRez:
        #     if i < theRez - 1 and not_edge[i]:
        #         i += 1
        #         continue
        #     bin_arr = self.dat_sorted[i_prev:i + 1]
        #     self.bin_the_slice(bin_count, bin_arr)
        #     bin_count += 1
        #     i += 1

        # top = int(np.ceil(np.max(self.rad_sorted)))
        # print("Top : {}".format(top))
        # bin_edges = np.arange(top)  # or whatever
        # self.binMin = np.empty(bin_edges.size - 1)
        # self.binMax = np.empty(bin_edges.size - 1)
        # self.binMid = np.empty(bin_edges.size - 1)
        #
        # for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        #     data_in_bin = self.dat_sorted[(self.rad_sorted >= bin_start) * (self.rad_sorted < bin_end)]
        #     if np.any(data_in_bin):
        #         self.binMin[i], self.binMax[i], self.binMid[i] = np.percentile(data_in_bin, [2, 95, 50])
        #     else:
        #         self.binMin[i], self.binMax[i], self.binMid[i] = np.nan, np.nan, np.nan

        # Bin the intensities by radius
        # self.radial_bins = np.empty((self.rez, self.rez))
        # self.radial_bins.fill(np.nan)
        # radial_counter = np.zeros(self.rez, dtype=np.int)
        # proper_bin = np.asarray(np.floor(self.rad_sorted), dtype=np.int32)

        # for item, binInd in enumerate(proper_bin):
        #     self.radial_bins[binInd, radial_counter[binInd]] = self.dat_sorted[item]
        #     radial_counter[binInd] += 1

        # for ii in np.arange(self.rez):
        #     useItems = np.isin(proper_bin, )
        # self.binMin , self.binMax, self.binMid = self.nan_percentile(self.radial_bins, [2, 95, 50], axis=1)

    # def bin_prep(self):
    #     self.proper_bin = np.asarray(np.floor(self.rad_sorted), dtype=np.int32)
    #
    #     self.binBtm = np.empty(self.rez)
    #     self.binMin = np.empty(self.rez)
    #     self.binMax = np.empty(self.rez)
    #     self.binMid = np.empty(self.rez)
    #     self.binTop = np.empty(self.rez)
    #
    #     self.binBtm.fill(np.nan)
    #     self.binMin.fill(np.nan)
    #     self.binMax.fill(np.nan)
    #     self.binMid.fill(np.nan)
    #     self.binTop.fill(np.nan)

    # def bin_stats(self):
    #     self.bin_prep()
    #
    #     bin_list = []
    #     last = 0
    #     for ii, (binInd, dat) in enumerate(zip(self.proper_bin, self.dat_sorted)):
    #         if binInd != last:
    #             self.bin_the_slice(last, bin_list)
    #             bin_list = []
    #         bin_list.append(dat)
    #         last = binInd

    def bin_the_slice(self, bin_count, bin_list):
        bin_array = np.asarray(bin_list)
        finite = bin_array[np.isfinite(bin_array)]
        data_in_bin = finite[np.nonzero(finite - self.offset)]
        if len(data_in_bin) > 0:
            self.binMin[bin_count], self.binMax[bin_count], self.binMid[bin_count] = np.percentile(data_in_bin, [2, 95, 50])

    def nan_percentile2(self, arr, q, interpolation='linear'):
        # valid (non NaN) observations along the first axis
        valid_obs = np.sum(np.isfinite(arr))
        if valid_obs <= 0:
            return np.nan, np.nan, np.nan
        # replace NaN with maximum
        max_val = np.nanmax(arr)
        arr[np.isnan(arr)] = max_val
        # sort - former NaNs will move to the end
        arr = np.sort(arr)

        # loop over requested quantiles
        if type(q) is list:
            qs = []
            qs.extend(q)
        else:
            qs = [q]
        if len(qs) < 2:
            quant_arr = np.zeros(shape=(arr.shape[0]))
        else:
            quant_arr = np.zeros(shape=(len(qs), arr.shape[0]))

        result = []
        for i in range(len(qs)):
            quant = qs[i]
            # desired position as well as floor and ceiling of it
            k_arr = (valid_obs - 1) * (quant / 100.0)
            f_arr = np.floor(k_arr).astype(np.int32)
            c_arr = np.ceil(k_arr).astype(np.int32)
            fc_equal_k_mask = f_arr == c_arr

            if interpolation == 'linear':
                # linear interpolation (like numpy percentile) takes the fractional part of desired position
                floor_val = np.take(arr, f_arr) * (c_arr - k_arr)
                ceil_val = np.take(arr, c_arr) * (k_arr - f_arr)

                quant_arr = floor_val + ceil_val
                if fc_equal_k_mask:
                    quant_arr = np.take(arr, k_arr.astype(np.int32))  # if floor == ceiling take floor value


            elif interpolation == 'nearest':
                f_arr = np.around(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'lowest':
                f_arr = np.floor(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'highest':
                f_arr = np.ceiling(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            result.append(quant_arr)

        return result

    def nan_percentile(self, arr, q, interpolation='linear', axis=0):
        # valid (non NaN) observations along the first axis
        valid_obs = np.sum(np.isfinite(arr), axis=axis)
        # replace NaN with maximum
        max_val = np.nanmax(arr)
        arr[np.isnan(arr)] = max_val
        # sort - former NaNs will move to the end
        arr = np.sort(arr, axis=axis)

        # loop over requested quantiles
        if type(q) is list:
            qs = []
            qs.extend(q)
        else:
            qs = [q]
        if len(qs) < 2:
            quant_arr = np.zeros(shape=(arr.shape[0], arr.shape[1]))
        else:
            quant_arr = np.zeros(shape=(len(qs), arr.shape[0], arr.shape[1]))

        result = []
        for i in range(len(qs)):
            quant = qs[i]
            # desired position as well as floor and ceiling of it
            k_arr = (valid_obs - 1) * (quant / 100.0)
            f_arr = np.floor(k_arr).astype(np.int32)
            c_arr = np.ceil(k_arr).astype(np.int32)
            fc_equal_k_mask = f_arr == c_arr

            if interpolation == 'linear':
                # linear interpolation (like numpy percentile) takes the fractional part of desired position
                floor_val = np.take(arr, f_arr) * (c_arr - k_arr)
                ceil_val = np.take(arr, c_arr) * (k_arr - f_arr)

                quant_arr = floor_val + ceil_val
                quant_arr[fc_equal_k_mask] = np.take(arr, k_arr.astype(np.int32))[fc_equal_k_mask]

            elif interpolation == 'nearest':
                f_arr = np.around(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'lowest':
                f_arr = np.floor(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)
            elif interpolation == 'highest':
                f_arr = np.ceiling(k_arr).astype(np.int32)
                quant_arr = np.take(arr, f_arr)

            result.append(quant_arr)

        return result

    def fit_curves(self):
        # Input Stuff
        self.radAbss = np.arange(self.rez)
        self.highCut = 730 * self.extra_rez
        theMin = 380 * self.extra_rez
        near_limb = np.arange(theMin, theMin + 50 * self.extra_rez)

        # Find the derivative of the binned Mid
        self.diff_Mid = np.diff(self.binMid)
        self.diff_Mid += np.abs(np.nanmin(self.diff_Mid))
        self.diff_Mid /= np.nanmean(self.diff_Mid) / 100

        # Locate the Limb
        self.limb_radii = np.argmin(self.diff_Mid[near_limb]) + theMin
        self.lCut = self.limb_radii - 10 * self.extra_rez
        self.hCut = self.limb_radii + 10 * self.extra_rez

        # Split into three regions
        self.low_abs = self.radAbss[:self.lCut]
        self.low_max = self.binMax[:self.lCut]
        self.low_min = self.binMin[:self.lCut]

        self.mid_abs = self.radAbss[self.lCut:self.hCut]
        self.mid_max = self.binMax[self.lCut:self.hCut]
        self.mid_min = self.binMin[self.lCut:self.hCut]

        self.high_abs = self.radAbss[self.hCut:]
        self.high_max = self.binMax[self.hCut:]
        self.high_min = self.binMin[self.hCut:]

        # Filter the regions separately
        from scipy.signal import savgol_filter

        lWindow = 20 * self.extra_rez + 1
        mWindow = 4 * self.extra_rez + 1
        hWindow = 30 * self.extra_rez + 1
        fWindow = int(3 * self.extra_rez) + 1

        rank = 3

        low_max_filt = savgol_filter(self.low_max, lWindow, rank)

        mid_max_filt = savgol_filter(self.mid_max, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)

        high_max_filt = savgol_filter(self.high_max, hWindow, rank)

        low_min_filt = savgol_filter(self.low_min, lWindow, rank)
        mid_min_filt = savgol_filter(self.mid_min, mWindow, rank)
        high_min_filt = savgol_filter(self.high_min, hWindow, rank)

        # Fit the low curves
        lmaxf = self.fill_start(low_max_filt)
        lminf = self.fill_start(low_min_filt)
        idx = np.isfinite(lmaxf)
        p = np.polyfit(self.low_abs[idx], lmaxf[idx], 9)
        low_max_fit = np.polyval(p, self.low_abs)
        p = np.polyfit(self.low_abs[idx], lminf[idx], 9)
        low_min_fit = np.polyval(p, self.low_abs)

        # Build output curves
        self.fakeAbss = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.fakeMax = np.hstack((low_max_fit, mid_max_filt, high_max_filt))
        self.fakeMin = np.hstack((low_min_fit, mid_min_filt, high_min_filt))

        # Filter again to smooth boundaraies
        self.fakeMax = self.fill_end(self.fill_start(savgol_filter(self.fakeMax, fWindow, rank)))
        self.fakeMin = self.fill_end(self.fill_start(savgol_filter(self.fakeMin, fWindow, rank)))

        # Locate the Noise Floor
        noiseMin = 550 * self.extra_rez - self.hCut
        near_noise = np.arange(noiseMin, noiseMin + 100 * self.extra_rez)
        self.diff_max_abs = self.high_abs[near_noise]
        self.diff_max = np.diff(high_max_filt)[near_noise]
        self.diff_max += np.abs(np.nanmin(self.diff_max))
        self.diff_max /= np.nanmean(self.diff_max) / 100
        self.noise_radii = np.argmin(self.diff_max) + noiseMin + self.hCut
        self.noise_radii = 565 * self.extra_rez

    def fill_end(self, use):
        iii = -1
        val = use[iii]
        while np.isnan(val):
            iii -= 1
            val = use[iii]
        use[iii:] = val
        return use

    def fill_start(self, use):
        iii = 0
        val = use[iii]
        while np.isnan(val):
            iii += 1
            try:
                val = use[iii]
            except:
                return use
        use[:iii] = val
        return use

    def plot_stats(self):

        fig, (ax0, ax1) = plt.subplots(2, 1, True)
        ax0.scatter(self.n2r(self.rad_sorted[::30]), self.dat_sorted[::30], c='k', s=2)
        ax0.axvline(self.n2r(self.limb_radii), ls='--', label="Limb")
        ax0.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        ax0.axvline(self.n2r(self.lCut), ls=':')
        ax0.axvline(self.n2r(self.hCut), ls=':')
        # ax0.axvline(self.tRadius, c='r')
        ax0.axvline(self.n2r(self.highCut))

        # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
        # plt.plot(self.radAbss[:-1] + 0.5, self.diff_Mid, 'r:')

        ax0.plot(self.n2r(self.low_abs), self.low_max, 'm', label="Percentile")
        ax0.plot(self.n2r(self.low_abs), self.low_min, 'm')
        # plt.plot(self.low_abs, self.low_max_fit, 'r')
        # plt.plot(self.low_abs, self.low_min_fit, 'r')

        ax0.plot(self.n2r(self.high_abs), self.high_max, 'c', label="Percentile")
        ax0.plot(self.n2r(self.high_abs), self.high_min, 'c')

        ax0.plot(self.n2r(self.mid_abs), self.mid_max, 'y', label="Percentile")
        ax0.plot(self.n2r(self.mid_abs), self.mid_min, 'y')
        # plt.plot(self.high_abs, self.high_min_fit, 'r')
        # plt.plot(self.high_abs, self.high_max_fit, 'r')

        try:
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.fakeAbss), self.fakeMin, 'g')
        except:
            ax0.plot(self.n2r(self.radAbss), self.fakeMax, 'g', label="Smoothed")
            ax0.plot(self.n2r(self.radAbss), self.fakeMin, 'g')

        # plt.plot(radAbss, binMax, 'c')
        # plt.plot(self.radAbss, self.binMin, 'm')
        # plt.plot(self.radAbss, self.binMid, 'y')
        # plt.plot(radAbss, binMed, 'r')
        # plt.plot(self.radAbss, self.binMax, 'b')
        # plt.plot(radAbss, fakeMin, 'r')
        # plt.ylim((-100, 10**3))
        # plt.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        ax0.set_xlim((0, self.n2r(self.highCut)))
        ax0.legend()
        fig.set_size_inches((8, 12))
        ax0.set_yscale('log')

        ax1.scatter(self.n2r(self.rad_flat[::10]), self.dat_coronagraph[::10], c='k', s=2)
        ax1.set_ylim((-0.25, 2))

        ax1.axhline(self.vmax, c='r', label='Confinement')
        ax1.axhline(self.vmin, c='r')
        ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
        ax1.axhline(self.vmin_plot, c='orange')

        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))

        ax1.legend()
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax0.set_ylabel(r"Absolute Intensity (Counts)")

        plt.tight_layout()
        if self.params.is_debug():
            file_name = '{}_Radial.png'.format(self.name)
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)

            file_name = '{}_Radial_zoom.png'.format(self.name)
            ax0.set_xlim((0.9, 1.1))
            save_path = join(self.params.local_directory, file_name)
            plt.savefig(save_path)
            # plt.show()
            plt.close(fig)
        else:
            plt.show()

    def n2r(self, n):
        if True:
            return n / self.limb_radii
        else:
            return n

