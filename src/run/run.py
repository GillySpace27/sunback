import sys

# # Main Command Structure
from time import sleep


class Runner:
    def __init__(self, params):
        self.params = params
    
    def start(self):
        """Select whether to run or to debug"""
        self.print_header()
        
        if self.params.is_debug():
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
                    out_string="I failed, but I'm ignoring it. Count: {}/{}\n".format(fail_count, fail_max)
                    print(out_string, error, "\n\n")
                    continue
                else:
                    print("Too Many Failures, I Quit!")
                    sys.exit(1)
            if self.params.stop_after_one():
                break
    
    def __process(self):
        """Use the provided fetcher, executor,
        and putter to do the thing"""
        print("Starting Batch: {}\n".format(self.params.batch_name()))
        if len(self.params.fetchers()) > 0:
            sys.stdout.flush()
            print(" >>>>>>>>>> Fetching Images <<<<<<<<<<", flush=True)
            for fet in self.params.fetchers():
                sleep(0.1)
                fet.fetch(self.params)
                sleep(0.1)
            
        if len(self.params.processors()) > 0:
            sys.stdout.flush()
            print(" >>>>>>>>>> Processing Images <<<<<<<<<<", flush=True)
            sys.stdout.flush()
            for proc in self.params.processors():
                sleep(0.1)
                proc.process(self.params)
                sleep(0.1)
                
        if len(self.params.putters()) > 0:
            sys.stdout.flush()
            print(" >>>>>>>>>> Outputting Images or Movies <<<<<<<<<<", flush=True)
            for put in self.params.putters():
                sleep(0.1)
                put.put(self.params)
                sleep(0.1)
        
        self.print_end_banner()
    

    ## PRINTING
    def print_header(self):
        print("\n\n*****************************************************************")
        print("\nSunback SDO Image Manipulator \nWritten by Chris R. Gilly")
        print("Check out my website: http://gilly.space\n")
        if self.params.is_debug(): print("DEBUG MODE\n")
        self.print_plan()
        print("\n*****************************************************************\n\n")
        
   
    def print_plan(self):
        print("Run Type: {}".format(self.params.run_type()))
        print("Here's the Plan:")
        if len(self.params.fetchers()) > 0:
            for fet in self.params.fetchers():
                fet.plan()
                
        if len(self.params.processors()) > 0:
            for proc in self.params.processors():
                proc.plan()
                
        if len(self.params.putters()) > 0:
            for put in self.params.putters():
                put.plan()
                
        print("  And Stop After One Loop" if self.params.stop_after_one() else "  And then repeat!")
        # print("\n")
    
    
        
    def print_end_banner(self):
        mode_string = "" if self.params.stop_after_one() else ", Restarting Loop"
        print("\n_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_")
        print("Program Complete{}".format(mode_string))
        print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n\n")