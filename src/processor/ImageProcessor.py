from processor.Processor import Processor


class ImageProcessor(Processor):
    
    filt_name = '  Video Writer'
    out_name = "_raw.avi"
    do_png = True
    wave = None
    progress_stem = "    Writing Movie {}"
    progress_text = ""
    video_name_stem = ""
    description = "Turn all the imgs into an AVI video"

def fits_to_pngs(self):
    """Re-save all the Fits images into pngs and normed fits files"""
    "Converting to Png Images..."
    self.apply_func_to_directory(self.do_image_work, doAll=False, desc=">Processing Images", unit="images")




def modify_img_series(self):
    """Processes the img series"""
    img_paths = []

    # paths = get_paths(self.params.local_fits_paths(),
    #                self.params.use_wavelengths, self.params.download_path())
    # self.params.local_fits_paths(paths)
    # self.params.local_fits_paths()

    for full_path in tqdm(self.params.local_fits_paths()):
        self.done_paths = find_done_paths(full_path)
        name = basename(full_path).casefold().replace("fits", "png")
        if name in self.done_paths and not self.params.overwrite_pngs():
            one_path = full_path
        else:
            with fits.open(full_path) as hdul:
            # try:
                one_path = self.modify_img(hdul, full_path)
            # except [TypeError(), IndexError()] as e:
            #     skipped += 1
            #     print(e)
            #     continue
        if type(one_path) not in [list]:
            one_path = [one_path]
        img_paths.extend(one_path)
        # break
    self.params.local_img_paths(img_paths)



    def modify_img(self, hdul, path=None):
        """modifies and uploads the in_object"""
        hdul.verify('silentfix+warn')

        save_path = path.replace("fits", "png")
        filename = basename(save_path)
        if filename.casefold() in self.done_paths:
            if not self.params.overwrite_pngs():
                return save_path
        try:
            hh = 0
            wave, t_rec = hdul[hh].header['WAVELNTH'], hdul[hh].header['T_OBS']
        except:
            hh = 1
            wave, t_rec = hdul[hh].header['WAVELNTH'], hdul[hh].header['T_OBS']
            # center = [hdul[hh].header[], hdul[hh].header[]]

        center = [hdul[hh].header['X0_MP'], hdul[hh].header['Y0_MP']]

        data = hdul[hh].data

        # Reduce the size of the array
        resolution = data.shape[0]
        desired = self.params.resolution()

        if resolution > desired:
            reduce_amount = int(resolution / desired)
            data = block_reduce(data, reduce_amount)
            center[0] /= reduce_amount
            center[1] /= reduce_amount

        # while center[0] > 0.9 * desired:
        #     center[0] /= 2
        #     center[1] /= 2


        # image_meta = str(wave), str(wave), t_rec, data.shape
        image_meta = str(wave), save_path, t_rec, data.shape

        img_paths = Modify(data, image_meta, center=center).get_paths()
        self.make_thumbs(img_paths[0])
        return img_paths



    def plot_and_save(self):

         self.render()

         self.export_files()

     def render(self):
         """Generate the plots"""
         image = self.changed
         original_image = self.original

         full_name, save_path, time_string, ii = self.image_data
         time_string2 = self.clean_time_string(time_string)
         name, wave = self.clean_name_string(full_name)

         self.figbox = []
         for processed in [False, True]:
             if not self.do_orig:
                 if not processed:
                     continue
             # Create the Figure
             fig, ax = plt.subplots()
             self.blankAxis(ax)
             fig.set_facecolor("k")

             self.inches = 10
             fig.set_size_inches((self.inches, self.inches))

             if 'hmi' in name.casefold():
                 inst = ""
                 plt.imshow(image, origin='upper', interpolation=None)
                 # plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)
                 plt.tight_layout(pad=5.5)
                 height = 1.05

             else:

                 # from .color_tables import aia_wave_dict
                 # aia_wave_dict(wave)

                 inst = '  AIA'
                 cmap = 'sdoaia{}'.format(wave)
                 cmap = aia_color_table(int(wave) * u.angstrom)
                 if processed:
                     plt.imshow(image, cmap=cmap, origin='lower', interpolation=None, vmin=self.vmin_plot, vmax=self.vmax_plot)
                 else:
                     toprint = self.normalize(self.absqrt(original_image))
                     # plt.imshow(toprint, cmap='sdoaia{}'.format(wave), origin='lower', interpolation=None) #,  vmin=self.vmin_plot, vmax=self.vmax_plot)

                     plt.imshow(self.absqrt(original_image), cmap=cmap, origin='lower', interpolation=None)  # ,  vmin=self.vmin_plot, vmax=self.vmax_plot)

                 plt.tight_layout(pad=0)
                 height = 0.95

             # Annotate with Text
             buffer = '' if len(name) == 3 else '  '
             buffer2 = '    ' if len(name) == 2 else ''

             title = "{}    {} {}, {}{}".format(buffer2, inst, wave, time_string2, buffer)
             ax.annotate(title, (0.15, height + 0.02), xycoords='axes fraction', fontsize='large',
                         color='w', horizontalalignment='center')
             # title2 = "{} {}, {}".format(inst, name, time_string2)
             # ax.annotate(title2, (0, 0.05), xycoords='axes fraction', fontsize='large', color='w')
             the_time = strftime("%Z %I:%M%p")
             if the_time[0] == '0':
                 the_time = the_time[1:]
             ax.annotate(the_time, (0.15, height), xycoords='axes fraction', fontsize='large',
                         color='w', horizontalalignment='center')

             # Format the Plot and Save
             self.blankAxis(ax)
             self.figbox.append([fig, ax, processed])
             if self.show:
                 plt.show()

     def export(self):
         full_name, save_path, time_string, ii = self.image_data
         pixels = self.changed.shape[0]
         dpi = pixels / self.inches
         try:
             self.img_box = []
             for fig, ax, processed in self.figbox:
                 # middle = '' if processed else "_orig"
                 #
                 # new_path = save_path[:-5] + middle + ".png"
                 # name = self.clean_name_string(full_name)
                 # directory = "renders/"
                 # path = directory + new_path.rsplit('/')[1]
                 # os.makedirs(directory, exist_ok=True)
                 # plt.close(fig)
                 # self.newPath = path

                 # Image from plot
                 ax.axis('off')
                 fig.tight_layout(pad=0)
                 # To remove the huge white borders
                 ax.margins(0)
                 ax.set_facecolor('k')

                 fig.canvas.draw()

                 image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                 image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                 self.img_box.append(image_from_plot)
                 # fig.savefig(path, facecolor='black', edgecolor='black', dpi=dpi)
                 # print("\tSaved {} Image:{}".format('Processed' if processed else "Unprocessed", name))
         except Exception as e:
             raise e
         finally:
             for fig, ax, processed in self.figbox:
                 plt.close(fig)

     def export_files(self):
         full_name, save_path, time_string, ii = self.image_data
         pixels = self.changed.shape[0]
         dpi = pixels / self.inches
         self.pathBox = []
         try:
             for fig, ax, processed in self.figbox:
                 middle = '' if processed else "_orig"


                 name, wave = self.clean_name_string(full_name)

                 save_directory = os.path.dirname(save_path)
                 if "fits" in save_directory:
                     save_directory = os.path.join(save_directory, "fits")
                 else:
                     save_directory = os.path.join(save_directory, "renders\\")

                 new_path = os.path.join(save_directory, name + middle + ".png")

                 if 'aia' in save_path:
                     os.makedirs(dirname(save_path), exist_ok=True)
                     new_path = save_path
                 else:
                     os.makedirs(save_directory, exist_ok=True)
                 fig.savefig(new_path, facecolor='black', edgecolor='black', dpi=dpi)
                 # print("\tSaved {} Image:{}".format('Processed' if processed else "Unprocessed", name))
                 self.pathBox.append(new_path)
         except Exception as e:
             raise e
         finally:
             for fig, ax, processed in self.figbox:
                 plt.close(fig)
             if False:
                 self.save_concatinated()

     def save_concatinated(self):
         name = self.pathBox[1][:-4] + "_cat.png"
         fmtString = "ffmpeg -i {} -i {} -y -filter_complex hstack {} -hide_banner -loglevel warning"
         os.system(fmtString.format(self.pathBox[1], self.pathBox[0], name))

         def make_png_path(self, fits_path):
             save_path = fits_path.replace("fits", "png")
             return basename(save_path)
     png_path = self.make_png_path(fits_path)

     if png_path.casefold() in self.done_paths:
         if not self.params.overwrite_pngs():
             return save_path

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

@staticmethod
def clean_name_string(full_name):
    digits = ''.join(i for i in full_name if i.isdigit())
    # Make the name strings
    name = digits + ''
    digits = "{:04d}".format(int(name))
    # while name[0] == '0':
    #     name = name[1:]
    return digits, name

@staticmethod
def clean_time_string(time_string):
    # Make the name strings
    
    cleaned = datetime.datetime.strptime(time_string[:-4], "%Y-%m-%dT%H:%M:%S")
    cleaned += timedelta(hours=-7)
    
    # tz = timezone(timedelta(range_hours=-1))
    # import pdb; pdb.set_trace()
    # cleaned = time_string.replace(tzinfo=timezone.utc).astimezone(tz=None)
    # cleaned = Time(time_string).datetime.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime("%I:%M%p, %b-%d, %Y")
    # cleaned = Time(time_string).datetime.replace(tzinfo=timezone.utc).astimezone(tz=tz).strftime("%I:%M%p, %b-%d, %Y")
    # cleaned = Time(time_string).datetime.strftime("%I:%M%p, %b-%d, %Y")
    # print("----------->", cleaned)
    # import pdb; pdb.set_trace()
    return cleaned.strftime("%m-%d-%Y %I:%M%p")
    # name = full_name + ''
    # while name[0] == '0':
    #     name = name[1:]
    # return name

@staticmethod
def absqrt(image):
    return np.sqrt(np.abs(image))


    # image_meta = str(wave), str(wave), t_rec, data.shape

    # self.make_thumbs(img_paths[0])
    # return img_paths
         if False:
             frame, center = reduce_array(frame, center, self.params.resolution())