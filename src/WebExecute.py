import sys
from platform import system
from os.path import join, abspath
from time import time


# Web Version
class WebExecute:
    def __init__(self, params):
        self.params = params
    
    def execute(self):
        # self.get()
        self.run()
    
    def get(self):
        """Download the images if there are new ones"""
        self.download_all_objects_in_aws_folder()
    
    def run(self):
        """Loop over the wavelengths and normalize, set background, and wait"""
        
        for file_path in self.download_all_objects_in_aws_folder():
            self.params.start_time = time()
            name = file_path[-8:-4]
            if self.params.do_one() and self.params.do_one() not in name:
                continue
    
            print("Image: {} at {}".format(name, file_path))
    
            # Update the Background
            self.update_background(file_path)
    
            if self.params.stop_after_one():
                sys.exit()
                
            # Wait for a bit
            self.params.sleep_until_delay_elapsed()
    
    
            print('')
    
    def download_all_objects_in_aws_folder(self):
        import boto3
        import os
        s3_resource = boto3.resource('s3')
        my_bucket = s3_resource.Bucket('gillyspace27-test-billboard')
        objects = my_bucket.objects.filter(Prefix='renders/')
        local_dir = self.params.discover_best_default_directory()
        # print("Save Path: {}".format(local_dir))
        print("Downloading to {}".format(local_dir))
        fileBox = []
        for obj in objects:
            path, filename = os.path.split(obj.key)
            if 'orig' in obj.key or 'archive' in obj.key or "thumbs" in obj.key or "4500" in obj.key:
                continue
            if self.params.do_one() and self.params.do_one() not in obj.key:
                continue
            print('    ', filename)
            loc = join(local_dir, filename)
            my_bucket.download_file(obj.key, loc)
            fileBox.append(loc)
        print("All Downloads Complete\n\n")
        return fileBox
        
        # # local_time_path = abspath(local_dir+r"/times.txt")
        # # local_fileBox_path = abspath(local_dir +r'/fileBox.dat')
        #
        # # Retrieve the file names
        # # web_path = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"
        # web_path = "http://jsoc2.stanford.edu/data/aia/synoptic/mostrecent/"
        #
        # # import pdb; pdb.set_trace()
        #
        # # local_path = abspath(r"C:\Users\chgi7364\Dropbox\AB_Interesting_Stuff\Projects\sunback_proj\sunback\data\images\times.txt")
        # self.fileBox = []
        #
        # #Find the time of the previous images
        # try:
        #     with open(local_time_path) as fp:
        #         header = fp.readline()
        #         _, old_datetime = header.split()
        # except:
        #     old_datetime = '20200101_000000'
        #
        # # Find the time of the newest images
        # print("Checking for New Images...", end='', flush=True)
        # urlretrieve(web_path + "image_times", local_time_path)
        #
        # with open(local_time_path) as fp:
        #     line = fp.readline()
        #     name, now = line.split()
        #     self.time_stamp = now
        #
        #     # Decide if new images are required
        #     there_arent_images = now <= old_datetime
        #     if there_arent_images or not self.params.download_images():
        #         # Use old images
        #         self.new_images = False
        #         try:
        #             with open(local_fileBox_path, 'r') as fp2:
        #                 for line in fp2:
        #                     a, b = line.split()
        #                     self.fileBox.append([a,b])
        #             print("None found!\n", flush=True)
        #
        #             need = False
        #             for label, file in self.fileBox:
        #                 if exists(file):
        #                     pass
        #                 else:
        #                     need = True
        #             if len(self.fileBox) == 0: need = True
        #             if not need:
        #                 return self.fileBox
        #             else:
        #                 print("Images Missing!\n", flush=True)
        #         except FileNotFoundError:
        #             print("New Images Required")
        #
        #     if False:
        #         print("Skipping!")
        #         return self.fileBox
        #
        #     # Get new images
        #     print("New images found!\n", flush=True)
        #     self.new_images = True
        #
        #     labels = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]
        #     import urllib
        #
        #     for name in tqdm(labels, unit="img", desc="Downloading Images", total=len(labels)):
        #
        #         # Ingest new images
        #         label = "{:04d}".format(int(name))
        #         webfile_name = r"AIAsynoptic{}.fits".format(label)
        #         directory_path = local_dir
        #         local_path = directory_path + r"/{}_MR.fits".format(label)
        #
        #         tries = 3
        #         for ii in np.arange(tries):
        #             try:
        #                 urlretrieve(web_path+webfile_name, local_path)
        #                 break
        #             except urllib.error.ContentTooShortError:
        #                 print("Failed Download...Retrying {} / {}".format(ii, tries))
        #                 pass
        #
        #
        #         self.fileBox.append([label, local_path])
        #     used = []
        #     # self.fileBox = list(set(self.fileBox))
        #     self.fileBox = [x for x in self.fileBox if x not in used and (used.append(x) or True)]
        #     self.fileBox = sorted(self.fileBox, key=lambda x: x[0])
        #     with open(local_fileBox_path, 'w') as fp:
        #         for a,b in self.fileBox:
        #             fp.write('{} {}\n'.format(a,b))
        # return self.fileBox
    
    @staticmethod
    def update_background(local_path, test=False):
        """
        Update the System Background

        Parameters
        ----------
        local_path : str
            The local save location of the image
            :param local_path:
            :param test:
        """
        local_path = abspath(local_path)
        # print(local_path)
        assert isinstance(local_path, str)
        print("Updating Background...", end='', flush=True)
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
        #
        # if self.params.is_debug():
        #     self.plot_stats()
        
        return 0
