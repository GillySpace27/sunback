"""
sunback.py
A program that downloads recent png images from Gilly's S3 bucket and sets them to the desktop background

Handles the primary functions
"""

# Imports
import matplotlib.image as mpimg
from time import localtime, timezone, strftime, time
from sunback.science.modify import Modify
from urllib.request import urlretrieve
from os import rename, remove
from os.path import normpath, abspath, join, exists
from calendar import timegm


from sunpy.net import Fido, attrs

# import sunpy.map
from threading import Barrier

from tqdm import tqdm
from platform import system
import sys
import numpy as np
import matplotlib as mpl
from sunback.science.parameters import Parameters

try:
    mpl.use("qt5agg")
except Exception as e:
    print(e)

import matplotlib.pyplot as plt

plt.ioff()

bbb = Barrier(3, timeout=100)

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

# print("Import took {:0.2f} seconds".format(time() - start))


def tr():
    import pdb

    pdb.set_trace()


class Sunback:
    """
    The Primary Class that Does Everything

    Parameters
    ----------
    parameters : Parameters (optional)
        a class specifying run options
    """

    def __init__(self, parameters):
        """Initialize a new parameter object or use the provided one"""
        self.indexNow = 0
        if parameters:
            self.params = parameters
        else:
            self.params = Parameters()
        self.last_time = 0
        self.this_time = 1
        self.new_images = True
        self.fido_result = None
        self.fido_num = None
        self.renew_mask = True
        self.mask_num = [1, 2]

    # MR Version

    def mr_execute(self):
        """Download the images and run the algorithm on them"""
        self.mr_get()
        self.mr_run()

    def mr_get(self):
        """Download the images if there are new ones"""
        local_dir = self.params.discover_best_default_directory()
        local_time_path = abspath(local_dir + r"/times.txt")
        local_fileBox_path = abspath(local_dir + r"/fileBox.dat")

        # Retrieve the file names
        web_path = "http://jsoc1.stanford.edu/data/aia/synoptic/mostrecent/"
        # import pdb; pdb.set_trace()

        # local_path = abspath(r"C:\Users\chgi7364\Dropbox\AB_Interesting_Stuff\Projects\sunback_proj\sunback\data\images\times.txt")
        self.fileBox = []

        # Find the time of the previous images
        try:
            with open(local_time_path) as fp:
                header = fp.readline()
                _, old_datetime = header.split()
        except:
            old_datetime = "20200101_000000"

        # Find the time of the newest images
        print("Checking for New Images...", end="", flush=True)
        urlretrieve(web_path + "image_times", local_time_path)

        with open(local_time_path) as fp:
            line = fp.readline()
            name, now = line.split()
            self.time_stamp = now

            # Decide if new images are required
            there_arent_images = now <= old_datetime

            if there_arent_images or not self.params.download_images():
                # Use old images
                self.new_images = False
                try:
                    with open(local_fileBox_path, "r") as fp2:
                        for line in fp2:
                            a, b = line.split()
                            self.fileBox.append([a, b])
                    print("None found!\n", flush=True)

                    need = False
                    for label, file in self.fileBox:
                        if exists(file):
                            pass
                        else:
                            need = True
                    if len(self.fileBox) == 0:
                        need = True
                    if not need:
                        return self.fileBox
                    else:
                        print("Images Missing!\n", flush=True)
                except FileNotFoundError:
                    print("New Images Required")

            # print("Skipping!")
            # return self.fileBox

            # Get new images
            print("New images found!\n", flush=True)
            self.new_images = True

            labels = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]
            import urllib

            for name in tqdm(
                labels, unit="img", desc="Downloading Images", total=len(labels)
            ):
                # Ingest new images
                label = "{:04d}".format(int(name))
                webfile_name = r"AIAsynoptic{}.fits".format(label)
                directory_path = local_dir
                local_path = directory_path + r"/{}_MR.fits".format(label)

                tries = 3
                for ii in np.arange(tries):
                    try:
                        urlretrieve(web_path + webfile_name, local_path)
                        break
                    except urllib.error.ContentTooShortError:
                        print("Failed Download...Retrying {} / {}".format(ii, tries))
                        pass

                self.fileBox.append([label, local_path])
            used = []
            # self.fileBox = list(set(self.fileBox))
            self.fileBox = [
                x for x in self.fileBox if x not in used and (used.append(x) or True)
            ]
            self.fileBox = sorted(self.fileBox, key=lambda x: x[0])
            with open(local_fileBox_path, "w") as fp:
                for a, b in self.fileBox:
                    fp.write("{} {}\n".format(a, b))
        return self.fileBox

    def mr_run(self):
        """Loop over the wavelengths and normalize, set background, and wait"""

        for this_name, file_path in self.fileBox:
            self.params.start_time = time()
            self.name = this_name

            if self.params.do_one() and self.params.do_one() not in this_name:
                continue

            print("Image: {}".format(this_name))

            # Modify the Image
            new_path = self.mr_modify(this_name, file_path)

            # Wait for a bit
            self.params.sleep_until_delay_elapsed()

            # Update the Background
            self.update_background(new_path)

            if self.params.stop_after_one():
                sys.exit()

            print("")

    def mr_modify(self, this_name, file_path):
        """Load the image, modify it, then plot and save"""
        print("Generating Image...", flush=True)

        # Open the File
        originalData, image_data = self.mr_open(this_name, file_path)

        # Modify the data
        mod_obj = Modify(originalData, image_data)
        new_path = mod_obj.get_path()
        return new_path

    def mr_open(self, this_name, file_path):
        """Load the Fits File from disk"""
        from astropy.io import fits

        for ind in np.arange(4):
            try:
                # Parse Inputs
                with fits.open(file_path) as hdul:
                    hdul.verify("silentfix+warn")

                    wave = hdul[0].header["WAVELNTH"]
                    t_rec = hdul[0].header["T_OBS"]
                    data = hdul[0].data
                    # print(wave, t_rec)

                    # plt.imshow(np.log10(img))
                    # plt.title("{}\n{}".format(wave, t_rec))
                    # plt.show()

                image_data = this_name, file_path, t_rec, 0
                break
            except TypeError as e:
                if ind < 3:
                    pass
                else:
                    raise e
        return data, image_data

    # Jp Version

    def jp_execute(self):
        self.jp_get()
        self.jp_run()

    def jp_get(self):
        # Retrieve the file names
        web_path = "http://jsoc1.stanford.edu/data/aia/images/image_times"
        local_path = "times.txt"
        self.fileBox = []

        # Find the time of the last images
        try:
            with open(local_path) as fp:
                header = fp.readline()
                _, old_datetime = header.split()
        except:
            old_datetime = "20200101_000000"

        # Find the time of the newest images
        urlretrieve(web_path, local_path)
        number_of_lines = len(open(local_path).readlines()) - 1

        with open(local_path) as fp:
            line = fp.readline()
            name, now = line.split()
            self.time_stamp = now
            print("Checking for New Images...", end="", flush=True)
            if now <= old_datetime and self.params.download_images():
                self.new_images = False
                try:
                    with open("data/images/fileBox.dat", "r") as fp2:
                        for line in fp2:
                            a, b = line.split()
                            self.fileBox.append([a, b])
                    print("None found!\n", flush=True)

                    need = False
                    for label, file in self.fileBox:
                        if exists(file[:-3] + "png"):
                            pass
                        else:
                            need = True
                    if len(self.fileBox) == 0:
                        need = True
                    if not need:
                        return self.fileBox
                    else:
                        print("Images Missing!\n", flush=True)
                except FileNotFoundError:
                    print("New Images Required")

            print("New images found!\n", flush=True)
            self.new_images = True

            for line in tqdm(
                fp, unit="img", desc="Downloading Images", total=number_of_lines
            ):
                name, value = line.split()

                # Ingest new images
                directory_path = "data/images"
                label = "{:04d}".format(int(name))
                file_name = "{}/AIA_{}.jp2".format(directory_path, label)
                urlretrieve(value, file_name)

                self.fileBox.append([label, file_name])
            used = []
            # self.fileBox = list(set(self.fileBox))
            self.fileBox = [
                x for x in self.fileBox if x not in used and (used.append(x) or True)
            ]
            self.fileBox = sorted(self.fileBox, key=lambda x: x[0])
            with open("data/images/fileBox.dat", "w") as fp:
                for a, b in self.fileBox:
                    fp.write("{} {}\n".format(a, b))
        return self.fileBox

    def jp_run(self):
        for this_name, file_path in self.fileBox:
            self.params.start_time = time()
            self.name = this_name
            if self.params.do_one() and self.params.do_one() not in this_name:
                continue

            print("Image: {}".format(this_name))

            # Modify the Image
            new_path = self.jp_modify(this_name, file_path)

            # Wait for a bit
            self.params.sleep_until_delay_elapsed()

            # Update the Background
            self.update_background(new_path)

            if self.params.stop_after_one():
                sys.exit()

            print("")

    def jp_open(self, this_name, file_path):
        from PIL import Image

        originalData = np.asarray(Image.open(file_path))
        save_path = file_path[:-3] + ".png"
        time_string = self.time_stamp
        image_data = this_name, save_path, time_string, 0
        return originalData, image_data

    def jp_modify(self, this_name, file_path):
        if True:
            print("Generating Image...", flush=True)

            # Open the File
            originalData, image_data = self.jp_open(this_name, file_path)

            # Modify the data
            data = self.image_modify(originalData)
            # data = originalData

            # Plot the Data
            new_path = self.plot_and_save(data, image_data, originalData)

        else:
            new_path = file_path[:-4] + ".png"

        return new_path

    def save_raw(self, label, file_name):
        from PIL import Image

        data = Image.open(file_name)
        save_path = file_name[:-3] + ".png"
        name = self.clean_name_string(label)
        time_string = self.time_stamp

        # Create the Figure
        fig, ax = plt.subplots()
        self.blankAxis(ax)

        inches = 10
        fig.set_size_inches((inches, inches))

        pixels = data.size[0]
        dpi = pixels / inches

        if "hmi" in name.casefold():
            inst = ""
            plt.imshow(data, origin="upper", interpolation=None)
            # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
            plt.tight_layout(pad=5.5)
            height = 1.05

        else:
            inst = "  AIA"
            plt.imshow(
                data,
                cmap="sdoaia{}".format(name),
                origin="lower",
                interpolation=None,
                vmin=self.vmin_plot,
                vmax=self.vmax_plot,
            )
            plt.tight_layout(pad=0)
            height = 0.95

        # Annotate with Text
        buffer = "" if len(name) == 3 else "  "
        buffer2 = "    " if len(name) == 2 else ""

        title = "{}    {} {}, {}{}".format(buffer2, inst, name, time_string, buffer)
        title2 = "{} {}, {}".format(inst, name, time_string)
        ax.annotate(
            title,
            (0.125, height + 0.02),
            xycoords="axes fraction",
            fontsize="large",
            color="w",
            horizontalalignment="center",
        )
        # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
        the_time = strftime("%I:%M%p").lower()
        if the_time[0] == "0":
            the_time = the_time[1:]
        ax.annotate(
            the_time,
            (0.125, height),
            xycoords="axes fraction",
            fontsize="large",
            color="w",
            horizontalalignment="center",
        )

        # Format the Plot and Save
        self.blankAxis(ax)
        middle = "" if False else "_orig"
        new_path = save_path[:-5] + middle + ".png"

        try:
            plt.savefig(new_path, facecolor="black", edgecolor="black", dpi=dpi)
            # print("\tSaved {} Image".format('Processed' if False else "Unprocessed"))
        except PermissionError:
            new_path = save_path[:-5] + "_b.png"
            plt.savefig(new_path, facecolor="black", edgecolor="black", dpi=dpi)
            print("Success")
        except Exception as e:
            print("Failed...using Cached")
            if self.params.is_debug():
                raise e
        plt.close(fig)

        return new_path

    # Fido Version, Level 1

    def fido_execute(self):
        # if self.params.time_period():
        self.fido_range()
        # else:
        # self.fido_search()

        while self.indexNow < self.fido_num:
            self.indexNow += 1
            self.fido_run(self.indexNow - 1)
        if self.params.do_HMI():
            self.indexNow = 0
            self.run_HMI()
            self.run_HMI_white()
        self.indexNow = 0

    def fido_run(self, ii):
        """The Main Loop"""

        # Gather Data + Print
        this_name = self.fido_get_name_by_index(ii)

        # if '4500' in this_name:
        #     return
        # if this_name in ['0094', '0131'] and self.params.is_first_run:
        #     # print("Skip for Now\n")
        #     return
        #
        # # if int(this_name) < 1000:
        # #     return
        #
        # if self.params.is_debug():
        #     if this_name in ['1600', '1700']:
        #         return

        # if this_name not in ['304']:
        #     return
        print(this_name, ii)
        # if self.params.do_171() and "171" not in this_name:
        #     return
        # if self.params.do_304() and "304" not in this_name:
        #     return
        if self.params.do_one() and self.params.do_one() not in this_name:
            return

        print("Image: {}".format(this_name))

        # Retrieve and Prepare the Image
        image_path = self.make_image(ii)

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        self.update_background(image_path)

        if self.params.stop_after_one():
            sys.exit()

        print("")

    def main_loop(self, ii):
        """The Main Loop"""

        # Gather Data + Print
        this_name = self.fido_get_name_by_index(ii)

        # if '4500' in this_name:
        #     return
        # if this_name in ['0094', '0131'] and self.params.is_first_run:
        #     # print("Skip for Now\n")
        #     return
        #
        # # if int(this_name) < 1000:
        # #     return
        #
        # if self.params.is_debug():
        #     if this_name in ['1600', '1700']:
        #         return
        #
        # # if this_name not in ['211']:
        # #     return

        if this_name not in ["304"]:
            return

        print("Image: {}".format(this_name))

        # Retrieve and Prepare the Image
        # image_path = self.make_image(ii)
        image_path = self.make_movie(ii)

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        # self.update_background(image_path)

        print("")

    def run_HMI(self):
        """The Secondary Loop"""

        web_path = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIBC.jpg"

        self.params.start_time = time()

        print("Image: {}".format("HMIBC"))
        # Define the Image
        self.hmi_path = normpath(join(self.params.local_directory, "HMIBC_Now.jpg"))

        # Download the Image
        self.download_image(self.hmi_path, web_path)

        # Modify the Image
        print("Modifying Image...", end="")
        new_path = self.plot_and_save(
            mpimg.imread(self.hmi_path),
            ("HMI", self.hmi_path, "Magnetic Field", -1),
            None,
        )

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        self.update_background(new_path)

        print("")

    def run_HMI_white(self):
        """The Secondary Loop"""

        web_path = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIIF.jpg"

        self.params.start_time = time()

        print("Image: {}".format("HMIF"))
        # Define the Image
        self.hmi_path = normpath(join(self.params.local_directory, "HMIF_Now.jpg"))

        # Download the Image
        self.download_image(self.hmi_path, web_path)

        # Modify the Image
        print("Modifying Image...", end="")
        new_path = self.plot_and_save(
            mpimg.imread(self.hmi_path), ("HMI", self.hmi_path, "White Light", -1)
        )

        # Wait for a bit
        self.params.sleep_until_delay_elapsed()

        # Update the Background
        self.update_background(new_path)

        print("")

    def download_image(self, local_path, web_path):
        """
        Download an image and save it to file

        Go to the internet and download an image

        Parameters
        ----------
        web_path : str
            The web location of the image

        local_path : str
            The local save location of the image
        """
        tries = 3

        for ii in range(tries):
            try:
                print("Downloading Image...", end="", flush=True)
                urlretrieve(web_path, local_path)
                print("Success", flush=True)
                return 0
            except KeyboardInterrupt:
                raise
            except Exception as exp:
                print("Failed {} Time(s).".format(ii + 1), flush=True)
                if ii == tries - 1:
                    raise exp

    # # Level 2 ##

    def fido_range(self):
        """Find the a certain image"""
        self.renew_mask = True
        self.fido_num = 0
        tries = 0
        minute_range = 2000
        if self.params.time_period():
            early = self.params.time_period()
        else:
            fmt_str = "%Y.%m.%d_%H:%M/{}m@{}m".format(
                minute_range, int(minute_range / 2)
            )
            early = strftime(fmt_str, localtime(time() - minute_range * 60 + timezone))
            # print(early)
            # now = strftime(fmt_str, localtime(time() + timezone))

        import drms

        K = drms.Client()
        series = r"aia.lev1_euv_12s"
        segment = "image"
        # SO I CAN ONLY GET THE 4 DAY OLD VERSION HERE
        # SET THIS UP LIKE BELOW
        time_query = "2020.08.12_TAI/1d@1d"  # /1d@6h"
        wavelength = ""

        query_string = series + "[{}][{}]".format(time_query, wavelength)
        print(query_string)
        aiakeys = K.pkeys(series)
        kk, ss = K.query(query_string, key=aiakeys, seg=segment)
        if len(ss) > 0:
            list_of_files = ss[segment]
            print(list_of_files)
        else:
            print(ss)
        import pdb

        pdb.set_trace()

        # import pdb;
        # pdb.set_trace()
        # print(kk)
        # print(ss)
        # query_string = series + "[? WAVELNTH=171 ?][]"
        # query_string = series + "[2020.04.20_TAI/1d@6h][]"
        # available_AIA = K.series(r'aia')
        # available_HMI = K.series(r'hmi')
        # si = K.info(series)
        # segments = si.segments

        from astropy.io import fits

        for file in list_of_files:
            url = "http://jsoc.stanford.edu" + file
            ret = urlretrieve(url)
            # print(ret)
            # with fits.open(ret[0]) as hdul:
            hdul = fits.open(ret[0])
            hdul.verify("fix")
            wave = hdul[1].header["WAVELNTH"]
            t_rec = hdul[1].header["T_REC"]
            img = hdul[1].data
            print(wave, t_rec)

            plt.imshow(np.log10(img))
            plt.title("{}\n{}".format(wave, t_rec))
            plt.show()

    def drms_download(self):
        pass

        # fits.getdata(ret[0])

        # fits_file = fits.getdata(url)

        # import pdb; pdb.set_trace()

        # while True and self.params.download_images():
        #     if tries > max_tries:
        #         self.new_images = False
        #         break
        #     try:
        #         # Define Time Range
        #         # fmt_str = '%Y/%m/%d %H:%M'
        #         # early = strftime(fmt_str, localtime(time() - minute_range * 60 + timezone))
        #         # now = strftime(fmt_str, localtime(time() + timezone))
        #         # override = False
        #         # if override:
        #         tries += 1
        #         if type(time_period) in [float, int, str]:
        #             early = time_period
        #             now = time_period + 10
        #         else:
        #             early = time_period[0] #'2020/05/26 00:00'
        #             now = time_period[1] #'2020/05/30 00:00'
        #         # Find Results
        #         self.fido_result = Fido.search(attrs.Time(early, now), attrs.Instrument('aia'))
        #         self.fido_num = self.fido_result.file_num
        #         self.new_images = True
        #         self.this_time = int(self.fido_result.get_response(0)[0].time.start)
        #
        #         break
        #         # else:
        #         #     # Find Results
        #         #     self.fido_result = Fido.search(attrs.Time(early, now), attrs.Instrument('aia'))
        #         #     self.fido_num = self.fido_result.file_num
        #         #     if self.params.is_debug():
        #         #         print(self.fido_num, '\t', minute_range)
        #         #     # Change time range if wrong number of records
        #         #     if self.fido_num > max_num:
        #         #         # tries += 1
        #         #         minute_range -= 5
        #         #         if tries > 3:
        #         #             if (tries - max_num) < 30:
        #         #                 continue
        #         #             minute_range -= 4
        #         #         continue
        #         #     if self.fido_num < min_num:
        #         #         tries += 1
        #         #         minute_range += 2
        #         #         if tries > 2:
        #         #             minute_range += 10
        #         #         if tries > 6:
        #         #             minute_range += 10
        #         #         continue
        #         #
        #         #     self.this_time = int(self.fido_result.get_response(0)[0].time.start)
        #         #     self.new_images = self.last_time < self.this_time
        #         #     break
        #     except ConnectionError:
        #         self.new_images = False
        #         break
        #
        # if self.new_images and self.params.download_images():
        #     print("Search Found {} new images at {}".format(self.fido_num,
        #                                                     self.parse_time_string_to_local(str(self.this_time), 2)), flush=True)
        #     self.last_time = self.this_time
        #
        #     with open(self.params.time_file, 'w') as fp:
        #         fp.write(str(self.this_time) + '\n')
        #         fp.write(str(self.fido_num) + '\n')
        #         fp.write(str(self.fido_result.get_response(0)))
        # else:
        #     print("No New Images, using Cached Data\n")
        #     self.fido_result = []
        #     self.new_images = False
        #
        #     with open(self.params.time_file, 'r') as fp:
        #         self.this_time = int(fp.readline())
        #         self.fido_num = int(fp.readline())
        #         fp.readline()
        #         fp.readline()
        #         fp.readline()
        #         while True:
        #             line = fp.readline()
        #             self.fido_result.append(line)
        #             if len(line) == 0:
        #                 break
        #         # print(self.fido_result)

    def fido_search(self):
        """Find the Most Recent Images"""
        self.renew_mask = True
        self.fido_num = 0
        tries = 0
        minute_range = 18

        max_tries = 20 if self.params.is_debug() else 10
        min_num = 3
        max_num = 15

        while True and self.params.download_images():
            if tries > max_tries:
                self.new_images = False
                break
            try:
                # Define Time Range
                fmt_str = "%Y/%m/%d %H:%M"
                early = strftime(
                    fmt_str, localtime(time() - minute_range * 60 + timezone)
                )
                now = strftime(fmt_str, localtime(time() + timezone))
                override = False
                if override:
                    early = "2020/05/26 00:00"
                    now = "2020/05/30 00:00"
                    # Find Results
                    self.fido_search_result = self.fido_result = Fido.search(
                        attrs.Time(early, now),
                        attrs.Instrument("aia"),
                        attrs.Resolution(self.params.resolution()),
                    )
                    self.fido_num = self.fido_result.file_num
                    self.new_images = True
                    self.this_time = int(self.fido_result.get_response(0)[0].time.start)

                    break
                else:
                    # Find Results
                    self.fido_result = Fido.search(
                        attrs.Time(early, now), attrs.Instrument("aia")
                    )
                    self.fido_num = self.fido_result.file_num
                    if self.params.is_debug():
                        print(self.fido_num, "\t", minute_range)
                    # Change time range if wrong number of records
                    if self.fido_num > max_num:
                        # tries += 1
                        minute_range -= 5
                        if tries > 3:
                            if (tries - max_num) < 30:
                                continue
                            minute_range -= 4
                        continue
                    if self.fido_num < min_num:
                        tries += 1
                        minute_range += 2
                        if tries > 2:
                            minute_range += 10
                        if tries > 6:
                            minute_range += 10
                        continue

                    self.this_time = int(self.fido_result.get_response(0)[0].time.start)
                    self.new_images = self.last_time < self.this_time
                    break
            except ConnectionError:
                self.new_images = False
                break

        if self.new_images and self.params.download_images():
            print(
                "Search Found {} new images at {}".format(
                    self.fido_num,
                    self.parse_time_string_to_local(str(self.this_time), 2),
                ),
                flush=True,
            )
            self.last_time = self.this_time

            with open(self.params.time_file, "w") as fp:
                fp.write(str(self.this_time) + "\n")
                fp.write(str(self.fido_num) + "\n")
                fp.write(str(self.fido_result.get_response(0)))
        else:
            print("No New Images, using Cached Data\n")
            self.fido_result = []
            self.new_images = False

            with open(self.params.time_file, "r") as fp:
                self.this_time = int(fp.readline())
                self.fido_num = int(fp.readline())
                fp.readline()
                fp.readline()
                fp.readline()
                while True:
                    line = fp.readline()
                    self.fido_result.append(line)
                    if len(line) == 0:
                        break
                # print(self.fido_result)

    # fp.write(str(self.this_time) + '\n')
    # fp.write(str(self.fido_num) + '\n')
    # fp.write(str(self.fido_result.get_response(0)))

    def fido_get_name_by_index(self, ind):
        self.params.start_time = time()
        if self.new_images:
            name = self.fido_result[0, ind].get_response(0)[0].wave.wavemin
        else:
            name = self.fido_result[ind][-6:-2]
            if name[-1] == ".":
                name = name[:-1]
            if name[0] == " ":
                name = name[1:]
        while len(name) < 4:
            name = "0" + name
        return name

    def make_movie(self, ii):
        # Download the fits data
        image_data = self.fido_download_by_index(ii)

        # Generate a png image
        image_path = self.fits_to_image(image_data)

        return image_path

    def make_image(self, ii):
        # Download the fits data
        image_data = self.fido_download_by_index(ii)

        # Generate a png image
        image_path = self.fits_to_image(image_data)

        return image_path

    # Level 3

    def fido_download_by_index(self, ind):
        """Retrieve a result by index and save it to file"""

        tries = 3

        if self.new_images and self.params.download_images():
            for ii in range(tries):
                try:
                    print("Downloading Fits Data...", end="", flush=True)
                    if self.useFido:
                        result = self.fido_retrieve_result(self.fido_result[0, ind])
                    else:
                        print("Success", flush=True)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exp:
                    print("Failed {} Time(s).".format(ii + 1), flush=True)
                    if ii == tries - 1:
                        return self.use_cached(ind)
        else:
            result = self.use_cached(ind)

        out = [x for x in result]
        out.append(ind)
        return out

    def fido_retrieve_result(self, this_result):
        """Retrieve a result and save it to file"""
        # Make the File Name
        name, save_path = self.get_paths(this_result)

        # Download and Rename the File
        time_string = self.fido_download(this_result, save_path)

        return name, save_path, time_string

    def fido_download(self, this_result, save_path):
        original = sys.stderr

        #  TODO make the error print to a log instead of console
        # sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w')
        downloaded_files = Fido.fetch(this_result, path=self.params.local_directory)
        # sys.stderr = original

        if exists(save_path):
            remove(save_path)
        rename(downloaded_files[0], save_path)

        try:
            time_string = self.parse_time_string_to_local(downloaded_files)
        except:
            time_string = "xxxx"

        return time_string

    def use_cached(self, ind):
        print("Using Cached Data...", end="", flush=True)
        result = self.list_files1(self.params.local_directory, "fits")
        file_name = [x for x in result][ind]
        full_name = file_name[:4]

        with open(self.params.time_file, "r") as fp:
            time_stamp = fp.read()
        time_string = self.parse_time_string_to_local(str(time_stamp), 2)
        save_path = join(self.params.local_directory, file_name)
        print("Success", flush=True)
        return full_name, save_path, time_string

    def fits_to_image(self, image_data):
        """Modify the Fits image into a nice png"""
        print("Generating Image...", flush=True)

        originalData, image_data = self.load_fits(image_data)

        # Modify the data
        data = self.image_modify(originalData)

        # Plot the Data
        new_path = self.plot_and_save(data, image_data, originalData)

        return new_path

    def load_fits(self, image_data):
        # Load the Fits File from disk
        full_name, save_path, time_string, ii = image_data

        for ind in np.arange(4):
            try:
                # Parse Inputs
                full_name, save_path, time_string, ii = image_data
                my_map = sunpy.map.Map(save_path)
                break
            except TypeError as e:
                if ind < 3:
                    image_data = self.fido_download_by_index(ii)
                else:
                    raise e
        data = my_map.data
        return data, image_data

    # Level 4

    @staticmethod
    def list_files1(directory, extension):
        from os import listdir

        return (f for f in listdir(directory) if f.endswith("." + extension))

    def get_paths(self, this_result):
        self.name = this_result.get_response(0)[0].wave.wavemin
        while len(self.name) < 4:
            self.name = "0" + self.name
        file_name = "{}_Now.fits".format(self.name)
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
            hour_raw = int(time_string[8:10])
            minute = time_string[10:12]

        hour = str(hour_raw % 12)
        if hour == "0":
            hour = 12
        suffix = "pm" if hour_raw > 12 else "am"
        struct_time = (
            int(year),
            int(month),
            int(day),
            hour_raw,
            int(minute),
            0,
            0,
            0,
            -1,
        )

        new_time_string = strftime(
            "%I:%M%p %m/%d/%Y ", localtime(timegm(struct_time))
        ).lower()
        if new_time_string[0] == "0":
            new_time_string = new_time_string[1:]

        # print(year, month, day, hour, minute)
        # new_time_string = "{}:{}{} {}/{}/{} ".format(hour, minute, suffix, month, day, year)
        return new_time_string

    @staticmethod
    def clean_name_string(full_name):
        # Make the name strings
        name = full_name + ""
        while name[0] == "0":
            name = name[1:]
        return name

    @staticmethod
    def clean_time_string(time_string):
        # Make the name strings
        from datetime import timezone
        from astropy.time import Time

        # import pdb; pdb.set_trace()
        # cleaned = time_string.replace(tzinfo=timezone.utc).astimezone(tz=None)
        cleaned = (
            Time(time_string)
            .datetime.replace(tzinfo=timezone.utc)
            .astimezone(tz=None)
            .strftime("%I:%M%p, %b-%d, %Y")
        )

        return cleaned
        # name = full_name + ''
        # while name[0] == '0':
        #     name = name[1:]
        # return name

    @staticmethod
    def blankAxis(ax):
        ax.patch.set_alpha(0)
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.tick_params(
            labelcolor="none",
            which="both",
            top=False,
            bottom=False,
            left=False,
            right=False,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Data Manipulations

    @staticmethod
    def reject_outliers(data):
        # # Reject Outliers
        # a = data.flatten()
        # remove_num = 20
        # ind = argpartition(a, -remove_num)[-remove_num:]
        # a[ind] = nanmean(a)*4
        # data = attrs.reshape(data.shape)

        data[data > 10] = np.nan

        return data

    @staticmethod
    def absqrt(data):
        return np.sqrt(np.abs(data))

    @staticmethod
    def normalize(data, high=98, low=15):
        if low is None:
            lowP = 0
        else:
            lowP = np.nanpercentile(data, low)
        highP = np.nanpercentile(data, high)

        return (data - lowP) / (highP - lowP)

    def vignette(self, data):
        mask = self.radius > (int(1.1 * self.rez // 2))  # (3.5 * self.noise_radii)
        data[mask] = np.nan
        return data

    def vignette2(self, data):
        mask = np.isclose(self.radius, self.tRadius, atol=2)
        data[mask] = 1

        mask = np.isclose(self.radius, self.noise_radii, atol=2)
        data[mask] = 1
        return data

    def coronagraph(self, data):
        # original = sys.stderr
        # sys.stderr = open(join(self.params.local_directory, 'log.txt'), 'w+')
        # print(data.dtype)
        data[data == 0] = np.nan
        radius_bin = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
        # import pdb; pdb.set_trace()
        dat_corona = (data.flatten() - self.fakeMin[radius_bin]) / (
            self.fakeMax[radius_bin] - self.fakeMin[radius_bin]
        )

        # sys.stderr = original

        # Deal with too hot things
        self.vmax = 0.95
        self.vmax_plot = 0.85  # np.max(dat_corona)
        hotpowr = 1 / 1.5

        hot = dat_corona > self.vmax
        # dat_corona[hot] = dat_corona[hot] ** hotpowr

        # Deal with too cold things
        self.vmin = 0.3
        self.vmin_plot = -0.05  # np.min(dat_corona)# 0.3# -0.03
        coldpowr = 1 / 2

        cold = dat_corona < self.vmin
        dat_corona[cold] = (
            -((np.abs(dat_corona[cold] - self.vmin) + 1) ** coldpowr - 1) + self.vmin
        )

        self.dat_coronagraph = dat_corona
        dat_corona_square = dat_corona.reshape(data.shape)

        if self.renew_mask or self.params.mode() == "r":
            self.corona_mask = self.get_mask(data)
            self.renew_mask = False

        dat_corona_square = dat_corona_square ** (1 / 5)
        data = self.normalize(data, high=100, low=0)
        dat_corona_square = self.normalize(dat_corona_square, high=100, low=1)

        # import pdb; pdb.set_trace()
        if self.params.do_mirror():
            # Do stuff
            xx, yy = self.corona_mask.shape[0], int(self.corona_mask.shape[1] / 2)
            # import pdb; pdb.set_trace()
            newDat = data[self.corona_mask]
            grid = newDat.reshape(xx, yy)
            # if self.
            flipped = np.fliplr(grid)
            data[~self.corona_mask] = flipped.flatten()  # np.flip(newDat)

        data[self.corona_mask] = dat_corona_square[self.corona_mask]
        # print(data.dtype)
        # import pdb; pdb.set_trace()
        # inds = np.argsort(self.rad_flat)
        # rad_sorted = self.rad_flat[inds]
        # dat_sort = dat_corona[inds]
        #
        # plt.figure()
        # # plt.yscale('log')
        # plt.scatter(rad_sorted[::30], dat_sort[::30], c='k')
        # plt.show()

        # data = data / np.mean(data)

        # data = data**(1/2)
        # data = np.log(data)

        # data = self.normalize(data, high=85, low=5)

        return data

    def get_mask(self, dat_out):
        corona_mask = np.full_like(dat_out, False, dtype=bool)
        rezz = corona_mask.shape[0]
        half = int(rezz / 2)

        mode = self.params.mode()

        if type(mode) in [float, int]:
            mask_num = mode
        elif "y" in mode:
            mask_num = 1
        elif "n" in mode:
            mask_num = 2
        else:
            if "r" in mode:
                if len(mode) < 2:
                    mode += "a"

            if "a" in mode:
                top = 8
                btm = 1
            elif "h" in mode:
                top = 6
                btm = 3
            elif "d" in mode:
                top = 8
                btm = 7
            elif "w" in mode:
                top = 2
                btm = 1
            else:
                print("Unrecognized Mode")
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
        self.offset = np.min(data)

        data -= self.offset

        self.make_radius(data)
        self.sort_radially(data)
        self.bin_radially()
        self.fit_curves()
        return data

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
        self.dat_sort = self.dat_flat[inds]

    def bin_radially(self):
        # Bin the intensities by radius
        self.radBins = [[] for x in np.arange(self.rez)]
        binInds = np.asarray(np.floor(self.rad_sorted), dtype=np.int32)
        for ii, binI in enumerate(binInds):
            self.radBins[binI].append(self.dat_sort[ii])

        # Find the statistics by bin
        self.binMax = np.zeros(self.rez)
        self.binMin = np.zeros(self.rez)
        self.binMid = np.zeros(self.rez)
        self.binMed = np.zeros(self.rez)
        self.radAbss = np.arange(self.rez)

        for ii, it in enumerate(self.radBins):
            item = np.asarray(it)

            idx = np.isfinite(item)
            finite = item[idx]
            idx2 = np.nonzero(finite - self.offset)
            subItems = finite[idx2]

            if len(subItems) > 0:
                self.binMax[ii] = np.percentile(subItems, 75)  # np.nanmax(subItems)
                self.binMin[ii] = np.percentile(subItems, 2)  # np.min(subItems)
                self.binMid[ii] = np.mean(subItems)
                self.binMed[ii] = np.median(subItems)
            else:
                self.binMax[ii] = np.nan
                self.binMin[ii] = np.nan
                self.binMid[ii] = np.nan
                self.binMed[ii] = np.nan

        # Remove NANs
        idx = np.isfinite(self.binMax) & np.isfinite(self.binMin)
        self.binMax = self.binMax[idx]
        self.binMin = self.binMin[idx]
        self.binMid = self.binMid[idx]
        self.binMed = self.binMed[idx]
        self.radAbss = self.radAbss[idx]

    def fit_curves(self):
        # Input Stuff
        self.highCut = 0.8 * self.rez

        # Locate the Limb
        theMin = int(0.35 * self.rez)
        theMax = int(0.45 * self.rez)
        near_limb = np.arange(theMin, theMax)

        r1 = self.radAbss[np.argmax(self.binMid[near_limb]) + theMin]
        r2 = self.radAbss[np.argmax(self.binMax[near_limb]) + theMin]
        r3 = self.radAbss[np.argmax(self.binMed[near_limb]) + theMin]

        self.limb_radii = int(np.mean([r1, r2, r3]))
        # print(self.limb_radii)
        self.lCut = int(self.limb_radii - 0.01 * self.rez)
        self.hCut = int(self.limb_radii + 0.01 * self.rez)

        # Split into three regions
        self.low_abs = self.radAbss[: self.lCut]
        self.low_max = self.binMax[: self.lCut]
        self.low_min = self.binMin[: self.lCut]

        self.mid_abs = self.radAbss[self.lCut : self.hCut]
        self.mid_max = self.binMax[self.lCut : self.hCut]
        self.mid_min = self.binMin[self.lCut : self.hCut]

        self.high_abs = self.radAbss[self.hCut :]
        self.high_max = self.binMax[self.hCut :]
        self.high_min = self.binMin[self.hCut :]

        if False:
            # plt.axvline(r1, c='g')
            # plt.axvline(r2, c='g')
            # plt.axvline(r3, c='g')

            plt.plot(self.radAbss, self.binMax, label="Max")
            plt.plot(self.radAbss, self.binMin, label="Min")
            plt.plot(self.radAbss, self.binMid, label="Mid")
            plt.plot(self.radAbss, self.binMed, label="Med")

            plt.axvline(theMin)
            plt.axvline(theMax)

            plt.axvline(self.limb_radii)
            plt.axvline(self.lCut, ls=":")
            plt.axvline(self.hCut, ls=":")
            plt.xlim([self.lCut, self.hCut])
            plt.legend()
            plt.show()

        # Filter the regions separately

        from scipy.signal import savgol_filter

        lWindow = 7  # 4 * self.extra_rez + 1
        mWindow = 7  # 4 * self.extra_rez + 1
        hWindow = 7  # 30 * self.extra_rez + 1
        fWindow = 7  # int(3 * self.extra_rez) + 1
        rank = 3

        # print(self.count_nan(self.throw_nan(self.low_max)))
        mode = "nearest"
        low_max_filt = savgol_filter(self.low_max, lWindow, rank, mode=mode)
        # import pdb; pdb.set_trace()

        mid_max_filt = savgol_filter(self.mid_max, mWindow, rank, mode=mode)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)
        # mid_max_filt = savgol_filter(mid_max_filt, mWindow, rank)

        high_max_filt = savgol_filter(self.high_max, hWindow, rank, mode=mode)

        low_min_filt = savgol_filter(self.low_min, lWindow, rank, mode=mode)
        mid_min_filt = savgol_filter(self.mid_min, mWindow, rank, mode=mode)
        high_min_filt = savgol_filter(self.high_min, hWindow, rank, mode=mode)

        # import pdb; pdb.set_trace()
        # Fit the low curves
        degree = 5
        # print(self.count_nan(low_max_filt))
        p = np.polyfit(self.low_abs, low_max_filt, degree)
        low_max_fit = np.polyval(p, self.low_abs)  # * 1.1
        p = np.polyfit(self.low_abs, low_min_filt, degree)
        low_min_fit = np.polyval(p, self.low_abs)

        ind = 10
        low_max_fit[0:ind] = low_max_fit[ind]
        low_min_fit[0:ind] = low_min_fit[ind]

        if False:
            plt.plot(self.low_abs, low_max_filt, lw=4)
            plt.plot(self.mid_abs, mid_max_filt, lw=4)
            plt.plot(self.high_abs, high_max_filt, lw=4)

            plt.plot(self.radAbss, self.binMax, label="Max")

            plt.plot(self.low_abs, low_min_filt, lw=4)
            plt.plot(self.mid_abs, mid_min_filt, lw=4)
            plt.plot(self.high_abs, high_min_filt, lw=4)

            plt.plot(self.radAbss, self.binMin, label="Min")

            plt.plot(self.low_abs, low_min_fit, c="k")
            plt.plot(self.low_abs, low_max_fit, c="k")

            # plt.plot(self.radAbss, self.binMid, label="Mid")
            # plt.plot(self.radAbss, self.binMed, label="Med")

            # plt.xlim([0.6*theMin,theMax*1.5])

            plt.legend()
            plt.show()

        # Build output curves
        self.fakeAbss = np.hstack((self.low_abs, self.mid_abs, self.high_abs))
        self.fakeMax0 = np.hstack((low_max_fit, mid_max_filt, high_max_filt))
        self.fakeMin0 = np.hstack((low_min_fit, mid_min_filt, high_min_filt))

        # Filter again to smooth boundaraies
        self.fakeMax0 = self.fill_end(
            self.fill_start(savgol_filter(self.fakeMax0, fWindow, rank))
        )
        self.fakeMin0 = self.fill_end(
            self.fill_start(savgol_filter(self.fakeMin0, fWindow, rank))
        )

        # Put the nans back in
        self.fakeMax = np.empty(self.rez)
        self.fakeMax.fill(np.nan)
        self.fakeMin = np.empty(self.rez)
        self.fakeMin.fill(np.nan)

        self.fakeMax[self.fakeAbss] = self.fakeMax0
        self.fakeMin[self.fakeAbss] = self.fakeMin0

        # plt.plot(np.arange(self.rez), self.fakeMax)
        # plt.plot(np.arange(self.rez), self.fakeMin)
        # plt.show()

        # # Locate the Noise Floor
        # noiseMin = 550 * self.extra_rez - self.hCut
        # near_noise = np.arange(noiseMin, noiseMin + 100 * self.extra_rez)
        # self.diff_max_abs = self.high_abs[near_noise]
        # self.diff_max = np.diff(high_max_filt)[near_noise]
        # self.diff_max += np.abs(np.nanmin(self.diff_max))
        # self.diff_max /= np.nanmean(self.diff_max) / 100
        # self.noise_radii = np.argmin(self.diff_max) + noiseMin + self.hCut
        # self.noise_radii = 565 * self.extra_rez

    def throw_nan(self, array):
        return array[~np.isnan(array)]

    def count_nan(self, array):
        return np.sum(np.isnan(array))

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
            val = use[iii]
        use[:iii] = val
        return use

    def plot_stats(self):
        fig, (ax0, ax1) = plt.subplots(2, 1, True)
        ax0.scatter(self.n2r(self.rad_sorted[::30]), self.dat_sort[::30], c="k", s=2)
        ax0.axvline(self.n2r(self.limb_radii), ls="--", label="Limb")
        # ax0.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
        ax0.axvline(self.n2r(self.lCut), ls=":")
        ax0.axvline(self.n2r(self.hCut), ls=":")
        # ax0.axvline(self.tRadius, c='r')
        ax0.axvline(self.n2r(self.highCut))

        # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
        # plt.plot(self.radAbss[:-1] + 0.5, self.diff_mean, 'r:')

        ax0.plot(self.n2r(self.low_abs), self.low_max, "m", label="Percentile")
        ax0.plot(self.n2r(self.low_abs), self.low_min, "m")
        # plt.plot(self.low_abs, self.low_max_fit, 'r')
        # plt.plot(self.low_abs, self.low_min_fit, 'r')

        ax0.plot(self.n2r(self.high_abs), self.high_max, "c", label="Percentile")
        ax0.plot(self.n2r(self.high_abs), self.high_min, "c")

        ax0.plot(self.n2r(self.mid_abs), self.mid_max, "y", label="Percentile")
        ax0.plot(self.n2r(self.mid_abs), self.mid_min, "y")
        # plt.plot(self.high_abs, self.high_min_fit, 'r')
        # plt.plot(self.high_abs, self.high_max_fit, 'r')

        # try:
        #     ax0.plot(self.n2r(self.fakeAbss), self.fakeMax, 'g', label="Smoothed")
        #     ax0.plot(self.n2r(self.fakeAbss), self.fakeMin, 'g')
        # except:
        #     ax0.plot(self.n2r(self.radAbss), self.fakeMax, 'g', label="Smoothed")
        #     ax0.plot(self.n2r(self.radAbss), self.fakeMin, 'g')

        # plt.plot(radAbss, binMax, 'c')
        # plt.plot(self.radAbss, self.binMin, 'm')
        # plt.plot(self.radAbss, self.binMid, 'y')
        # plt.plot(radAbss, binMed, 'r')
        # plt.plot(self.radAbss, self.binMax, 'b')
        # plt.plot(radAbss, fakeMin, 'r')
        # plt.ylim((-100, 10**3))
        # plt.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
        # ax0.set_xlim((0, self.n2r(self.highCut)))
        ax0.legend()
        fig.set_size_inches((8, 12))
        ax0.set_yscale("log")

        ax1.scatter(
            self.n2r(self.rad_flat[::10]), self.dat_coronagraph[::10], c="k", s=2
        )
        ax1.set_ylim((-0.25, 2))

        ax1.axhline(self.vmax, c="r", label="Confinement")
        ax1.axhline(self.vmin, c="r")
        ax1.axhline(self.vmax_plot, c="orange", label="Plot Range")
        ax1.axhline(self.vmin_plot, c="orange")

        # locs = np.arange(self.rez)[::int(self.rez/5)]
        # ax1.set_xticks(locs)
        # ax1.set_xticklabels(self.n2r(locs))

        ax1.legend()
        ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
        ax1.set_ylabel(r"Normalized Intensity")
        ax0.set_ylabel(r"Absolute Intensity (Counts)")

        plt.tight_layout()
        if True:  # self.params.is_debug():
            file_name = "{}_Radial.png".format(self.name)
            # print("Saving {}".format(file_name))
            save_path = join(r"data\images\radial", file_name)
            plt.savefig(save_path)

            file_name = "{}_Radial_zoom.png".format(self.name)
            ax0.set_xlim((0.9, 1.1))
            save_path = join(r"data\images\radial", file_name)
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


# Helper Functions


def run(delay=60, mode="y", debug=False, do171=False, do304=False):
    p = Parameters()
    p.set_delay_seconds(delay)
    p.do_171(do171)
    p.do_304(do304)

    if debug:
        p.is_debug(True)
        p.set_delay_seconds(10)
        p.do_HMI(False)

    # p.time_period(period=['2019/12/21 04:20', '2019/12/21 04:40'])
    p.resolution(2048)
    p.range(days=5)  # 0.060)
    p.download_images(True)
    p.cadence(3)
    p.frames_per_second(20)
    p.bpm(150)
    # p.download_images(False)
    # p.overwrite_pngs(False)
    p.sonify_limit(False)
    # p.remove_old_images(True)
    p.make_compressed(True)
    p.sonify_images(True, True)
    # p.sonify_images(False, False)
    # p._stop_after_one = True
    # p.do_171(True)
    # p.do_304(True)

    Sunback(p).start()


def where():
    """Prints the location that the images are stored in."""
    p = Parameters()
    print(p.discover_best_default_directory())


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    where()
    run(10, "y", debug=debugg)


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
