from time import sleep
from putter.Putter import Putter

class NullPutter(Putter):
    description = "Print the Name of the Download Folder"
    """Saves some data"""
    def put(self, params=None):
        self.load(params)
        # n_files = len(self.params.local_imgs_paths()) + len(self.params.local_fits_paths())
        print("  Files Saved in {}\n".format(self.params.base_directory()))
        # print("   No Output Selected\n")
        
        