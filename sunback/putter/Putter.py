from time import sleep
from tqdm import tqdm
from sunback.processor.Processor import Processor
import os.path

class Putter(Processor):
    """Saves some data"""
    description = "Use an Unnamed Putter"
    def __init__(self, params=None, quick=False, rp=None, in_name=None):
        super().__init__(params, quick, rp)

    def put(self, params=None):
        self.load(params)
        raise NotImplementedError()

    def process(self, params=None):
        self.put(params)
        # super().cleanup


    def sleep_until_delay_elapsed(self, delay=None):
        """ Make sure that the loop takes the right amount of time """
        delay = delay or self.params.delay_seconds()
        try:
            for ii in tqdm((range(int(delay))), ncols=120, desc=" *   {}, Waiting for {:0.0f} seconds".format(os.path.basename(self.png_name), delay)):
                sleep(1)
        except KeyboardInterrupt:
            # print("\rSkipping!")
            pass

            # if brk:
            #     break