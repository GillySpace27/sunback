from time import sleep
from putter.Putter import Putter

class NullPutter(Putter):
    description = "Print the Name of the Image Folder"
    filt_name = "NullPutter"
    """Saves some data"""
    def put(self, params=None):
        self.load(params)
        # n_files = len(self.params.local_imgs_paths()) + len(self.params.local_fits_paths())
        print(" {}\n".format(self.params.base_directory()))
        self.toc()
        # print("   No Output Selected\n")
        
        