from time import sleep

from tqdm import tqdm

from processor.Processor import Processor

class Putter(Processor):
    """Saves some data"""
    description = "Use an Unnamed Putter"
    
    def put(self, params=None):
        self.load(params)
        raise NotImplementedError()
    
    def process(self, params=None):
        self.put(params)
        self.toc()
        
    def sleep_until_delay_elapsed(self):
        """ Make sure that the loop takes the right amount of time """
        delay = self.params.delay_seconds()
        try:
            for ii in tqdm((range(int(delay))), ncols=120, desc=" *   {}, Waiting for {:0.0f} seconds".format(self.png_name, delay)):
                sleep(1)
        except KeyboardInterrupt:
            # print("\rSkipping!")
            pass
                
            # if brk:
            #     break