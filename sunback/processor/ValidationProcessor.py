# import os
# from os import makedirs
# from os.path import join, dirname
# import numpy as np
# from scipy.signal import savgol_filter
# from src.processor.Processor import Processor
#
# import warnings
#
# warnings.filterwarnings("ignore")
# import matplotlib as mpl
#
# mpl.use("qt5agg")
# import matplotlib.pyplot as plt
#
# plt.ioff()
#
# do_dprint = False
#
#
# def dprint(txt, **kwargs):
#     if do_dprint:
#         print(txt, **kwargs)
#
#
import os
from os.path import join

from astropy.io import fits
from tqdm import tqdm

from sunback.processor.Processor import Processor


class ValidationProcessor(Processor):
    filt_name = 'Validator'
    description = "Verify Good Images"
    progress_verb = "Validating"
    progress_unit = "Fits Files"

    def __init__(self, fits_path=None, in_name=-1, orig=False, show=False, verb=False, quick=False, rp=None, params=None):
        """Initialize the main class"""
        super().__init__(params, quick, rp)
#         # Parse Inputs
#
#     ###################
#     ## Structure ##
#     ###################
#
#     def setup(self):
#         """Do prep work once before the main algorithm"""
#         raise NotImplementedError
#
    def fetch(self):
        """Do whatever you want to each image_path in the directory"""
        print("Old validator tried to run --------------")

        pass
        # to_destroy = self.validate_fits()
        # self.destroy_files(to_destroy)
#                self.remove_files(local_fits_path)
#     def cleanup(self):
#         """Runs once after all the images have been modified with do_work"""
#         raise NotImplementedError
#
#     def do_fits_function(self, fits_path=None, in_name=None):
#         """Calls the do_work function on a single fits path if indicated"""
#         if self.load_fits_image(fits_path):
#             if (not self.use_keyframes) or (self.fits_path in self.keyframes):
#                 return self.do_work()  # Do the work on the fits files
#         return None
#
#     ###################
#     ## Top-Level ##
#     ###################
#
    # def validate_fits(self):
    #     import numpy as np
    #     if self.params.skip_validation:
    #         return []
    #     all_fits_paths = self.params.local_fits_paths()
    #     n_fits = len(all_fits_paths)
    #     destroyed = 0
    #     dark = 0
    #     missing = 0
    #     to_destroy = []
    #     print('Validation is Running')
    #     for local_fits_path in tqdm(all_fits_paths, desc=" > Validating Fits Files", unit='imgs'):
    #     # for local_fits_path in all_fits_paths:
    #         delete = False
    #
    #         with fits.open(local_fits_path, ignore_missing_end=True) as hdul:
    #             hdul.verify('silentfix+warn')
    #             self.rename_initial_frames(hdul) # This might not work
    #             #TEST 1 - IS IT A DARK FRAME?
    #             img_type = hdul[1].header['IMG_TYPE']
    #             if img_type.casefold() == 'dark'.casefold():
    #                 delete = True
    #                 dark += 1
    #                 print("Dark Image Detected")
    #
    #             #TEST 2 - IS FILLED WITH NULLS?
    #             frame = hdul[-1].data
    #             good_pix = np.sum(np.isfinite(frame))
    #             total_pix = frame.shape[0]*frame.shape[1]
    #             good_percent = good_pix / total_pix
    #             # Dark Frame had 0.37
    #             if good_percent < 0.6:
    #                 # print("Good Percent: {:0.4f}".format(good_percent))
    #                 delete = True
    #                 missing += 1
    #                 print("Missing Data Detected")
    #             hdul.close()
    #
    #         if delete:
    #             to_destroy.append(local_fits_path)
    #             destroyed += 1
    #             n_fits -= 1
    #     if destroyed:
    #         print(" ii     Fits Files Validated: {}, Bad Frames: {}\n".format(n_fits, destroyed))
    #     else:
    #         print(" ii     Fits Files Validated: {}. No Bad Frames!\n".format(n_fits))
    #
    #     if missing:
    #         print(" ii          > {} missing data".format(missing))
    #     if dark:
    #         print(" ii          > {} dark frames".format(dark))
    #
    #     return to_destroy

    def destroy_files(self, to_destroy=[]):
        if to_destroy:
            for path in to_destroy:
                self.remove_files(path)

    def remove_files(self, local_fits_path):
        # if local_fits_path in self.params.local_fits_paths():
        #     self.params.local_fits_paths().remove(local_fits_path)

        dir = os.path.dirname(local_fits_path)
        directory = dir.replace('fits', 'png\\mod')
        file = os.path.basename(local_fits_path)
        png_file = file.replace('.fits', '.png')
        png_path = os.path.join(directory, png_file)

        dead_paths = [local_fits_path, png_path, png_path.replace("mod", "cat"), png_path.replace("mod", "orig")]
        deleted_files = 0
        print()
        for path in dead_paths:
            try:
                print("    Deleting a File...", end="")
                os.remove(path)
                if os.path.exists(path):
                    raise FileExistsError(path)
                print("Success!")
                deleted_files += 1
            except PermissionError as e:
                print(" Couldn't Access/Delete\n {}".format(path))
                print(e)
                # raise e
            except FileExistsError as e:
                print(" File was still there after attempted deletion\n{}".format(os.path.basename(path)))
                # print(e)
            except FileNotFoundError as e:
                # print(" Not Found to Delete\n {}".format(path))
                pass
            except Exception as e:
                print(" Failed to Delete\n {}".format(path))
                print(e)
                1+1
        print("Actually Deleted {} Files".format(deleted_files))

        # try:
        # # try:
        # #     hh = 0
        # #     total_counts = np.nansum(hdul[hh].data)
        # # except Exception as e:
        # #     print(e)
        # #     hh = 1
        # #     delete = True
        # #     total_counts = np.nansum(hdul[hh].data)
        # #     delete = False
        # # this_size = stat(abs_path).st_size
        # # data = hdul[hh].data
        # # if total_counts < 0:  # or not this_size == self.file_size_mode:
        # #     delete = True
        # except TypeError as e:
        #     print(e)
        #     delete = True


#
#     def redownload_bad_fits(self):
#         if len(self.redownload) > 0:
#             self.redownload = []
#             self.fido_get_fits()
#
#
#
#     def remove_all_old_pngs(self):
#         requested_pngs = [x.replace('fits', 'png') for x in self.params.local_fits_paths()]
#         png_directory = join(self.params.imgs_top_directory(), self.params.current_wave(), 'png')
#         got_png = self.params.local_imgs_paths()  # list_files_in_directory(png_directory, 'png')
#         remove_count = 0
#         for mod_path in got_png:
#             if mod_path not in requested_pngs:
#                 try:
#                     os.remove(join(png_directory, mod_path))
#                     remove_count += 1
#                 except FileNotFoundError as e:
#                     # print(e)
#                     pass
#         if remove_count > 0:
#             print("{} old pngs deleted".format(remove_count))
#
#     def remove_all_old_fits_pngs(self):
#         keep = []
#         self.file_size = []
#         for local_file in self.params.local_fits_paths():
#             if local_file not in self.requested_files:
#                 pointing_start = self.parse_filename_to_time(local_file)
#                 if pointing_start not in self.requested_files:
#                     self.remove_fits_and_png(local_file)
#             else:
#                 keep.append(local_file)
#                 self.file_size.append(stat(join(self.fits_folder, local_file)).st_size)
#             self.params.local_fits_paths(keep)
#
#         if len(self.redownload) > 0:
#             print("        Deleting old files...", pointing_end='')
#             print("  Success! Deleted {} old images".format(len(self.redownload)))
#
#
#
#
#
#
#
#
#
#
#
#     ###################
#     ## Keyframes ##
#     ###################
#
#     def add_to_keyframes(self):
#         """Records the current analysis as one of the radial samples"""
#         self.update_keyframe_counters()
#         if not self.outputs_initialized:
#             self.init_running_curves()
#         else:
#             self.update_running_curves()
#
#
#     def update_keyframe_counters(self, n=1):
#         """Keep track of how many items have been added to keyframes"""
#         self.n_keyframes += n
#         self.lastIndex += n
#         self.skipped -= n
#         self.skipped = max(self.skipped, 0)
#
#     ######################################
#     ## Initializeing and Converting ##
#     ######################################
#
#     def init_for_learn(self):
#         self.init_images()
#         self.init_frame_curves()
#         self.init_radius_array()
#         self.init_statistics()
#
#     def init_for_modify(self):
#         self.init_radius_array()
#
#     def init_radius_array(self, vignette_radius=1.2, s_radius=400, t_factor=1.28, force=False):
#         """Build an r-coordinate array of shape(in_object)"""
#
#         if self.rez is None:
#             self.rez = self.modified_image.shape[0]
#             self.output_abscissa = np.arange(self.rez)
#
#         if self.radius is None or force or self.modified_image.shape[0] != self.rez:
#             dprint("init_radius_array")
#
#             xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
#             xc, yc = xx - self.center[0], yy - self.center[1]
#
#             self.radius = np.sqrt(xc * xc + yc * yc)
#             self.rad_flat = self.radius.flatten()
#             self.binInds = np.asarray(np.floor(self.rad_flat), dtype=np.int32)
#
#             self.vignette_mask = self.radius > (int(vignette_radius * self.rez // 2))
#             self.s_radius = s_radius
#             self.tRadius = self.s_radius * t_factor
#
#     def init_images(self, modified_image=None):
#         """Get all the variables ready for the normalization"""
#         dprint("\ninit_images")
#         if modified_image is not None:
#             self.modified_image = modified_image
#         self.rez = self.modified_image.shape[0]
#         self.modified_image = self.modified_image.astype('float32')
#         self.modified_image[self.modified_image == 0] = np.nan
#
#         if self.raw_image is None:
#             self.raw_image = self.modified_image + 0
#
#         self.raw_flat = self.raw_image.flatten()
#         self.changed_flat = self.modified_image.flatten()
#
#
#
#
#     ###################################
#     ## Normalization Curve Stuff ##
#     ###################################
#
#
#     def save_cached_data(self, radBins=None):
#         if radBins is not None:
#             self.radBins = radBins
#         self.save_frame_to_fits_file(fits_path=self.fits_path, frame=np.asarray(self.radBins), out_name='radBins')
#         pass
#
#     def load_cached_data(self, in_name='radBins'):
#         self.load_a_fits_attribute(fits_path=self.fits_path, in_name='radBins')
#         pass
#
#     def prep_save_outs(self):
#         """Prepare the scalar_out_curve for writing"""
#         if self.outer_min is None:
#             return None
#         self.scalar_out_curve = np.zeros(len(self.outer_min))
#         if self.limb_radius_from_header:
#             self.scalar_out_curve[0] = self.limb_radius_from_header
#         if self.abs_min_scalar:
#             self.scalar_out_curve[1] = self.abs_min_scalar
#             self.scalar_out_curve[2] = self.abs_max_scalar
#
#         out_list = [self.outer_min, self.inner_min, self.inner_max, self.outer_max, self.scalar_out_curve]
#         none_check = [item is not None for item in out_list]
#         self.do_save = np.all(none_check)
#         print("do save: ", self.do_save)
#         self.curve_out_array = np.asarray(out_list)
#         return self.do_save
#
#     def unpack_save_ins(self):
#         """Prepare the scalar_out_curve for writing"""
#         self.outer_min, self.inner_min, self.inner_max, \
#         self.outer_max, self.scalar_in_curve = np.loadtxt(self.params.curve_path())
#
#         self.limb_radius_from_header = self.scalar_in_curve[0]
#         self.abs_min_scalar = self.scalar_in_curve[1]
#         self.abs_max_scalar = self.scalar_in_curve[2]
#
#         # self.plot_norm_curves(show=True, save=False)
#
#
#
#     ####################################
#     ## Image Reduction Algorithms ##
#     ####################################
#
#     def prep_output(self):
#
#         self.mask_output()
#         self.mirror_output()
#
#         # Un-Flatten the Array
#         self.modified_image = self.changed_flat.reshape(self.modified_image.shape)
#         self.modified_image = np.sign(self.modified_image) * np.power(np.abs(self.modified_image), (1 / 5))
#         self.modified_image = self.modified_image.astype('float32')
#
#
#
#     ########################
#     ## Plotting Stuff ##
#     ########################
#
#     # def plot_all(self, do=True):
#     #     # self.plot_curves_2(do, False)
#     #     # self.plot_curves(do, False)
#     #     self.plot_full_normalization(do, False)
#     #     # self.QRNPlot()
#     #     pass
#     #     # plt.show()
#
#     def plot_curves_2(self, do=True, show=False):
#         if not do: return
#
#         fig, ax = plt.subplots()
#         ax.set_title("Plot Curves 2")
#
#         # Plot the filtered curves
#         ax.plot(self.low_abs, self.low_max_filt, lw=4)
#         ax.plot(self.mid_abs, self.mid_max_filt, lw=4)
#         ax.plot(self.high_abs, self.high_max_filt, lw=4)
#
#         ax.plot(self.binAbss, self.binMax, label="Max")
#
#         ax.plot(self.low_abs, self.low_min_filt, lw=4)
#         ax.plot(self.mid_abs, self.mid_min_filt, lw=4)
#         ax.plot(self.high_abs, self.high_min_filt, lw=4)
#
#         ax.plot(self.binAbss, self.binMin, label="Min")
#
#         ax.plot(self.low_abs, self.low_min_fit, c='k')
#         ax.plot(self.low_abs, self.low_max_fit, c='k')
#
#         ax.plot(self.output_abscissa, self.franken_max, label="FinalMax", lw=5)
#         ax.plot(self.output_abscissa, self.franken_min, label="FinalMin", lw=5)
#
#         # plt.plot(self.binAbss, self.binAbsMax, label="Mid")
#         # plt.plot(self.binAbss, self.binAbsMin, label="Med")
#
#         # plt.xlim([0.6*theMin,theMax*1.5])
#
#         ax.legend()
#         if show: plt.show()
#
#     def plot_curves(self, do=False, show=False):
#         """Plot the radial statistics from the binned array"""
#         if not do: return
#
#         fig, ax = plt.subplots()
#         ax.set_title("Plot Curves")
#
#         ax.plot(self.binAbss, self.binMax, label="Max")
#         ax.plot(self.binAbss, self.binMin, label="Min")
#         ax.plot(self.binAbss, self.binAbsMax, label="Mid")
#         ax.plot(self.binAbss, self.binAbsMin, label="Med")
#
#         ax.axvline(self.theMin)
#         ax.axvline(self.theMax)
#
#         ax.axvline(self.limb_radius_from_header)
#         ax.axvline(self.lCut, ls=':')
#         ax.axvline(self.hCut, ls=':')
#         ax.set_xlim([self.lCut, self.hCut])
#         ax.legend()
#         if show:
#             ax.show()
#
#     def plot_full_normalization(self, do=False, show=False, save=True, get_normed=False):
#         """This plot is in radius and has a scatter plot
#             overlaid with the norm curves as determined elsewhere"""
#         if not do:
#             return
#         dprint("plot_full_normalization")
#         # Init the Plots
#         fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
#
#         ## Plot the extrema curves
#         self.output_abscissa = np.arange(self.rez)
#         ax0.plot(self.n2r(self.output_abscissa), self.outer_max, label="OuterMax", lw=3, c='orange')
#         ax0.plot(self.n2r(self.output_abscissa), self.inner_max, label="InnerMax", lw=3, c='gold')
#         ax0.plot(self.n2r(self.output_abscissa), self.inner_min, label="InnerMin", lw=3, c='cornflowerblue')
#         ax0.plot(self.n2r(self.output_abscissa), self.outer_min, label="OuterMin", lw=3, c='b')
#         ax0.axvspan(self.n2r(self.limb_radius_from_header), ls='-', label="Limb")
#
#         # try:
#         #     ## Vertical Lines
#         #     ax0.axvline(self.n2r(self.lCut), ls=':')
#         #     ax0.axvline(self.n2r(self.hCut), ls=':')
#         #
#         #     # ax.axvline(self.n2r(self.noise_radii), c='r', ls='--', label="Scope Edge")
#         #     # ax0.axvline(self.n2r(self.highCut))
#         # except Exception as e:
#         #     print("QRNProc1::", e)
#         #     # raise e
#
#         ## Scatter Plot the intensities
#         skip = 50
#         ax0.scatter(self.n2r(self.rad_flat[::skip]), self.raw_image.flatten()[::skip], c='k', s=2)
#
#         if False:  # get_normed and self.modified_image.flatten() is None:
#             self.image_modify()
#             # if self.changed_flat is not None:
#             ax1.scatter(self.n2r(self.rad_flat[::10]), self.modified_image.flatten()[::10], c='k', s=2)
#
#         ## Plot Formatting
#         ax0.set_title("Plot Stats")
#         ax0.set_ylim((10 ** -2, 10 ** 4))
#         ax0.legend()
#         ax0.set_yscale('log')
#         ax0.set_ylabel(r"Absolute Intensity (Counts)")
#
#         # ax1.legend()
#         ax1.axhline(1)
#         ax1.axhline(0.05)
#         ax1.set_xlabel(r"Distance from Center of Sun ($R_\odot$)")
#         ax1.set_ylabel(r"Normalized Intensity")
#         ax1.set_yscale('log')
#         ax1.set_ylim((10 ** -2, 10 ** 2.5))
#
#         plt.tight_layout()
#         fig.set_size_inches(8, 12)
#
#         self.force_save_radial_figures(save, fig, ax0, show)
#
#         # ax.axvline(self.tRadius, c='r')
#
#         # plt.plot(self.diff_max_abs + 0.5, self.diff_max, 'r')
#         # plt.plot(self.binAbss[:-1] + 0.5, self.diff_mean, 'r:')
#
#         ## Norm Curves
#
#         # ax0.plot(self.n2r(self.low_abs), self.low_max, 'm', label="low_min/max")
#         # ax0.plot(self.n2r(self.low_abs), self.low_min, 'm', label="")
#         # # plt.plot(self.low_abs, self.low_max_fit, 'r')
#         # # plt.plot(self.low_abs, self.low_min_fit, 'r')
#         #
#         # ax0.plot(self.n2r(self.high_abs), self.high_max, 'c', label="high_min/max")
#         # ax0.plot(self.n2r(self.high_abs), self.high_min, 'c', label="")
#         #
#         # ax0.plot(self.n2r(self.mid_abs), self.mid_max, 'y', label="mid_min/max")
#         # ax0.plot(self.n2r(self.mid_abs), self.mid_min, 'y', label="")
#         # plt.plot(self.high_abs, self.high_min_fit, 'r')
#         # plt.plot(self.high_abs, self.high_max_fit, 'r')
#
#         # try:
#         #     ax0.plot(self.n2r(self.rendered_abss), self.outer_min, label="FinalMax", lw=4, c='blue')
#         #     ax0.plot(self.n2r(self.rendered_abss), self.outer_max, label="FinalMin", lw=4, c='orange')
#         # except Exception as e:
#         #     print("QRNProc2::", e)
#         #     # raise e
#
#         # try:
#         #     ax1.plot(self.n2r(self.output_abscissa), self.frame_maximum, 'g', label="Smoothed")
#         #     ax1.plot(self.n2r(self.output_abscissa), self.frame_minimum, 'g')
#         # except:
#         #     ax1.plot(self.n2r(self.binAbss), self.frame_maximum, 'g', label="Smoothed")
#         #     ax1.plot(self.n2r(self.binAbss), self.frame_minimum, 'g')
#
#         # ax1.plot(binAbss, binMax, 'c')
#         # ax1.plot(self.n2r(self.binAbss), self.binMin, 'm')
#         # ax1.plot(self.n2r(self.binAbss), self.binAbsMax, 'y')
#         # ax1.plot(self.n2r(self.binAbss), self.binMax, 'b')
#         # ax1.plot(binAbss, binAbsMin, 'r')
#         # ax1.plot(binAbss, frame_minimum, 'r')
#         # ax1.set_ylim((-0.5, 2))
#         # ax1.xlim((380* self.extra_rez ,(380+50)* self.extra_rez ))
#         # ax1.set_xlim((0, self.n2r(self.highCut)))
#
#         # ax1.axhline(self.vmax, c='r', label='Confinement')
#         # ax1.axhline(self.vmin, c='r')
#         # ax1.axhline(self.vmax_plot, c='orange', label='Plot Range')
#         # ax1.axhline(self.vmin_plot, c='orange')
#
#         # locs = np.arange(self.rez)[::int(self.rez/5)]
#         # ax1.set_xticks(locs)
#         # ax1.set_xticklabels(self.n2r(locs))
#
#         return True
#
#     def force_save_radial_figures(self, save, fig, ax0, show):
#         first = True
#         while True:
#             try:
#                 self.save_radial_figures(save, fig, ax0, show)
#                 # if not first: print("  Thanks, good job.\n")
#                 break
#             except OSError as e:
#                 if first:
#                     print("\n\n", e)
#                     print("  !!!!!!! Close the Dang Plot!", pointing_end='')
#                     first = False
#                 print('.', pointing_end='')
#
#     def save_radial_figures(self, do=False, fig=None, ax=None, show=False):
#         if not do: return
#         # print("Saving {}".format(file_name_1))
#         bs = self.params.base_directory()
#         folder_name = "radial"
#         file_name_1 = 'Radial_{}.png'.format(self.file_basename[:-5])
#         save_path_1 = join(bs, folder_name, file_name_1)
#         file_name_2 = 'zoom\\Radial_zoom_{}.png'.format(self.file_basename[:-5])
#         save_path_2 = join(bs, folder_name, file_name_2)
#
#         makedirs(dirname(save_path_1), exist_ok=True)
#         plt.savefig(save_path_1)
#
#         makedirs(dirname(save_path_2), exist_ok=True)
#         # ax.set_xlim((0.9, 1.1))
#         plt.savefig(save_path_2)
#
#         if not show:
#             plt.close(fig)
#         else:
#             plt.show()
#
#     def get_points(self, index):
#         ## Scatter Plot
#         skip = 100
#         return None
#         plotY = self.radBins_all[index]
#
#         xBox = []
#         yBox = []
#         for ii, bin in enumerate(plotY):
#             for item in bin:
#                 xBox.append(self.n2r(ii))
#                 yBox.append(item)
#
#         out = np.array((xBox, yBox))
#         return out
#         # return np.asarray(np.concatinate(xBox, yBox))
#
#     ########################
#     ## Utilities ##
#     ########################
#     ## Static Methods ##
#     def n2r(self, n):
#         """Convert index to solar radius"""
#         if n is None:
#             n = 0
#         return n / self.limb_radius_from_header
#
#     @staticmethod
#     def normalize(image_path, high=98, low=15):
#         """Normalize the Array"""
#         if low is None:
#             lowP = 0
#         else:
#             lowP = np.nanpercentile(image_path, low)
#         highP = np.nanpercentile(image_path, high)
#         import warnings
#         with warnings.catch_warnings():
#             warnings.filterwarnings('error')
#             try:
#                 out = (image_path - lowP) / (highP - lowP)
#             except RuntimeWarning as e:
#                 out = image_path
#         return out
#
#     @staticmethod
#     def norm_formula(flat_image, the_min, the_max):
#         """Standard Normalization Formula"""
#         top = np.subtract(flat_image, the_min)
#         bottom = np.subtract(the_max, the_min)
#         return np.divide(top, bottom)
#
#
#     @staticmethod
#     def fill_end(use):
#         iii = -1
#         val = use[iii]
#         while np.isnan(val):
#             iii -= 1
#             val = use[iii]
#         use[iii:] = val
#         return use
#
#     @staticmethod
#     def fill_start(use):
#         iii = 0
#         val = use[iii]
#         while np.isnan(val):
#             iii += 1
#             val = use[iii]
#         use[:iii] = val
#         return use
