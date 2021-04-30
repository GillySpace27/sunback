from execute.AwsExecutor import AwsExecutor
from execute.BackgroundExecutor import BackgroundExecutor
import sys
import sunback as sb
from traceback import print_tb


# # Main Command Structure

class Runner:
    def __init__(self, params):
        self.params = params
    
    def start(self):
        """Select whether to run or to debug"""
        self.__print_header()
        
        if self.params.is_debug():
            self.__debug_mode()
        else:
            self.__run_mode()
    
    def __print_header(self):
        print("\nSunback SDO Image Manipulator \nWritten by Chris R. Gilly")
        print("Check out my website: http://gilly.space\n")
        print("Delay: {} Seconds".format(self.params.delay_seconds()))
        # print("Coronagraph Mode: {} \n".format(params.mode()))
        
        if self.params.is_debug():
            print("DEBUG MODE\n")
    
    def __debug_mode(self):
        """Run the program in a way that will break"""
        while True:
            self.__execute()
    
    def __run_mode(self):
        """Run the program in a way that won't break"""
        
        fail_count = 0
        fail_max = 10
        
        while True:
            try:
                self.__execute()
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
    
    def __execute(self):
        """Use the provided fetcher and executor to do the thing"""
        
        # Get the paths to the files to be worked upon
        if self.params.fetcher() is not None:
            paths = self.params.fetcher().download_fits_files()
        else:
            paths = []
        self.params.executor().execute(paths)
