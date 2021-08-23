import sys


# # Main Command Structure
from utils.file_util import print_end_banner, print_header


class Runner:
    def __init__(self, params):
        self.params = params
    
    def start(self):
        """Select whether to run or to debug"""
        debug = self.params.is_debug()
        print_header(self.params.delay_seconds(),
                     self.params.download_path(), debug)
        
        if debug:
            self.__debug_mode()
        else:
            self.__run_mode()
    
    def __debug_mode(self):
        """Run the program in a way that will break"""
        while True:
            self.__process()
            if self.params.stop_after_one():
                break
    
    def __run_mode(self):
        """Run the program in a way that won't break"""
        
        fail_count = 0
        fail_max = 10
        
        while True:
            try:
                self.__process()
                fail_count -= 1
            except (KeyboardInterrupt, SystemExit):
                print("\n\nOk, I'll Stop. Doot!\n")
                break
            except Exception as error:
                fail_count += 1
                if fail_count < fail_max:
                    print("I failed, but I'm ignoring it. Count: {}/{}\n\n".format(fail_count, fail_max))
                    # print_tb(error)
                    continue
                else:
                    print("Too Many Failures, I Quit!")
                    sys.exit(1)
            if self.params.stop_after_one():
                break
    
    def __process(self):
        """Use the provided fetcher, executor,
        and putter to do the thing"""
        
        self.params.fetcher().fetch()
        
        print("Processing Images...", flush=True)
        for proc in self.params.processors():
            proc.process()

        self.params.putter().put()
        
        print_end_banner(self.params.stop_after_one())
    

