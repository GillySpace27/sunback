from time import sleep
from putter.Putter import Putter

class NullPutter(Putter):
    def __init__(self, params):
        self.params = params
        self.params.stop_after_one(True)
    """Saves some data"""
    def put(self):
        print("Doing Nothing With the Images\n")
        sleep(1)
        
        