from processor.Processor import Processor

class Putter(Processor):
    """Saves some data"""
    description = "Use an Unnamed Putter"
    
    def put(self, params=None):
        self.load(params)
        raise NotImplementedError()