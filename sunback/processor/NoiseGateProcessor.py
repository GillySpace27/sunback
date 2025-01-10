from os import listdir
from os.path import join
from sunback.processor.Processor import Processor
import noisegate.tools as ngt
import numpy as np
from tqdm import tqdm


class NoiseGateProcessor(Processor):
    """This class template holds the code for the Sunpy Processors"""
    name = filt_name = "Noise Gate Processor"
    description = "Apply Noise Gate filter to images"
    progress_verb = 'Gating'
    finished_verb = "Noise Gated"
    out_name = "Gated"
    in_name = "lev1p0"

    def do_work(self):
        """Analyze the Image, Normalize it, Plot"""
        if self.params.do_single:
            self.load_fits_images_single()
        else:
            raise NotImplementedError
        self.noise_gate()
        return self.params.modified_image

    def load_fits_images_single(self):
        print(r"Loading Data Cube of {} A...".format(self.params.current_wave()), end='')
        use_path = self.fits_path or self.params.use_image_path()
        raw_dir = use_path[:-5]

        fits_in_paths = [join(raw_dir, x) for x in listdir(raw_dir) if ".fits" in x]
        frames = []
        # pairs = []
        for path in tqdm(fits_in_paths, desc=" *    Loading Frames"):
            frame, wave, t_rec, center, int_time, name = self.load_this_fits_frame(path, self.in_name, quiet=True)
            frames.append(frame)
            # pairs.append((frame,header))

        self.n_loaded = len(frames)
        print(" *      Successfully loaded {} frames!".format(self.n_loaded))

        self.frameCube = np.asarray(frames, dtype=np.float64)
        # self.frameSequence = sunpy.map.Map(pairs, sequence=True)

    def noise_gate(self):
        print(" *    Beginning Noise Gating Procedure...", end='')
        out = ngt.noise_gate_batch(self.frameCube)  # , cubesize=12, model='hybrid', factor=2.0)
        self.params.modified_image = bigOut = np.sum(out, axis=0)
        print("Noise Gating Complete!")

        return bigOut

#
#
#
# class NoiseGateProcessor_old(Processor):
#
#     def __init__(self, params=None, rp=None, fits_path=None):
#         super().__init__(params=params, rp=rp)
#         self.fits_path = fits_path or self.fits_path or self.params.fits_path
#         self.params = params
#         self.in_name = ["lev1p0"]
#         self.local_wave_directory = None
#         self.image_folder = None
#         self.movie_folder = None
#         self.video_name_stem = None
#         self.wave = None
#
#
#     def do_work(self):
#         a=1
#         pass
#
#
#
#
#
#
#
#     #
#     #
#     # def process(self, params=None):
#     #     """loads fits images and then performs the noise gating on them"""
#     #     for self.wave in self.params.use_wavelengths:
#     #         if self.wave not in self.params.do_one():
#     #             continue
#     #         if self.params.do_single:
#     #             self.load_fits_images_single()
#     #         else:
#     #             raise NotImplementedError
#     #         self.noise_gate()
#     #         # self.save_cubes()
#
#     def load_fits_images_single(self):
#         print(r"Loading Data Cube of {} A...".format(self.wave), end='')
#         use_path = self.fits_path or self.params.use_image_path()
#         raw_dir = use_path[:-5]
#         self.set_in_frame_name(use_path)
#
#         fits_in_paths = [join(raw_dir, x) for x in listdir(raw_dir) if ".fits" in x]
#         frames = []
#         # pairs = []
#         for path in fits_in_paths:
#             frame, wave, t_rec, center, int_time, img_type, header = self.load_this_fits_frame(path, self.in_name, quiet=True)
#             frames.append(frame)
#             # pairs.append((frame,header))
#
#         self.n_loaded = len(frames)
#         print("Successfully loaded {} frames!".format(self.n_loaded))
#
#         self.frameCube = np.asarray(frames, dtype=np.float64)
#         # self.frameSequence = sunpy.map.Map(pairs, sequence=True)
#
#         # shape = self.frameCube.shape
#         # typ = self.frameCube.dtype
#         # a = 1
#
#     def noise_gate(self):
#         print("Beginning Noise Gating Procedure...", end='')
#         now = time.time()
#         out = ngt.noise_gate_batch(self.frameCube) #, cubesize=12, model='hybrid', factor=2.0)
#         self.params.modified_image = bigOut = np.sum(out, axis=0)
#         then = time.time() - now
#         print("Noise Gating Complete. Took:")
#         print("{:0.2} seconds, or".format(then))
#         print("{:0.2} minutes".format(then/60))
#         return bigOut
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
#
#
#
#
#
# max_use = 20
# n_chunks = int(self.number // max_use)
# for ii in tqdm(np.arange(n_chunks)):
#     start = (ii) * max_use
#     end = (ii + 1) * max_use
#     cubie = self.H_XY[start:end]
#     self.gated_cube[start:end] = out
#     break
#     if ii * max_use > small_fill:
#         break
#     # for jj in np.arange(start_timestamp=start_timestamp, stop=pointing_end):
#     #     # fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
#     #     # ax1.imshow(norm(self.H_XY[jj]))
#     #     # ax2.imshow(norm(self.gated_cube[jj]))
#     #     # plt.show(block=True)
#     #
#     #     fig, (ax) = plt.subplots(1, 1)
#     #     plt.title(jj)
#     #     ax.imshow(self.gated_cube[jj] - self.H_XY[jj])
#     #     plt.show(block=True)


#
#
#
#
#
#
#
#     # the_path = join(raw_path, '*.fits')
#     # mc = sunpy.map.Map(the_path, sequence=True)
#     print("Hello World")
#     a = 1+1
#     # self.prepare_to_load()
#     # self.allocate_cubes()
#     # self.fill_cubes()s
#     # print("Cube has been loaded!", flush=True)
#
# # def prepare_to_load(self):
# #     self.build_paths()
# #     self.sample_frame = self.load_file(self.im_paths[0])
# #     self.height, self.width = self.sample_frame.shape
# #     self.number = len(self.im_paths)
#
# def build_paths(self):
#     self.local_wave_directory = join(self.params.imgs_top_directory(), self.wave)
#     self.image_folder = join(self.local_wave_directory, 'png')
#     self.fits_folder = join(self.local_wave_directory, 'fits')
#     self.movie_folder = abspath(join(self.params.imgs_top_directory(), "movies\\"))
#     self.video_name_stem = join(self.movie_folder, '{}_{}_movie{}'.format(self.wave, strftime('%m%d_%H%M'), '{}'))
#     self.im_paths = [join(self.fits_folder, img) for img in listdir(self.fits_folder) if img.endswith(".fits")]
#     makedirs(self.movie_folder, exist_ok=True)
#
# def allocate_cubes(self):
#     try:
#         self.H_XY
#         self.gated_cube
#     except AttributeError:
#         self.H_XY = np.empty((self.number, self.width, self.height), dtype=type(self.sample_frame[0, 0]))
#         self.gated_cube = self.H_XY + 0
#     self.H_XY.fill(np.nan)
#     self.gated_cube.fill(np.nan)
#
# def fill_cubes(self):
#     for ii, img in enumerate(tqdm(self.im_paths)):
#         frame = self.load_file(img)
#         if frame is not None and frame.shape == self.sample_frame.shape:
#             self.H_XY[ii] = frame
#
#         if ii > small_fill:
#             break
#
# def save_cubes(self):
#     for ii, img in enumerate(tqdm(self.im_paths)):
#         self.save_file(img, self.H_XY[ii])
#         # self.load_file(img)
#         if ii > small_fill:
#             break
#
# def noise_gate(self):
#     print("Beginning Noise Gating Procedure...", end='')
#     max_use = 20
#     n_chunks = int(self.number // max_use)
#     for ii in tqdm(np.arange(n_chunks)):
#         start = (ii) * max_use
#         end = (ii + 1) * max_use
#         cubie = self.H_XY[start:end]
#         out = ngt.noise_gate_batch(cubie, cubesize=12, model='hybrid', factor=2.0)
#         self.gated_cube[start:end] = out
#         break
#         if ii * max_use > small_fill:
#             break
#         # for jj in np.arange(start_timestamp=start_timestamp, stop=pointing_end):
#         #     # fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)
#         #     # ax1.imshow(norm(self.H_XY[jj]))
#         #     # ax2.imshow(norm(self.gated_cube[jj]))
#         #     # plt.show(block=True)
#         #
#         #     fig, (ax) = plt.subplots(1, 1)
#         #     plt.title(jj)
#         #     ax.imshow(self.gated_cube[jj] - self.H_XY[jj])
#         #     plt.show(block=True)
#     print("Noise Gating Complete")
#
#     #
#     #     if len(images) > 0:
#     #
#     #
#     #
#     #
#     #         in_object = cv2.imread(join(self.image_folder, images[0]))
#     #         height, width, layers = in_object.shape
#     #         final_output_path = self.video_name_stem.format("_raw.avi")
#     #         print(final_output_path)
#     #         video_avi = cv2.VideoWriter(final_output_path, 0, self.params.frames_per_second(), (width, height))
#     #
#     #         for in_object in tqdm(images, desc=">Noise Gating {}".format(current_wave), unit="in_object"):
#     #             # print(join(self.image_folder, in_object))
#     #             im = cv2.imread(join(self.image_folder, in_object))
#     #             video_avi.write(im)
#     #
#     #         cv2.destroyAllWindows()
#     #         video_avi.release()
#     #
#     #     else:
#     #         print("No png Images Found")
#     # except FileNotFoundError:
#     #     print("Images Not Found")
