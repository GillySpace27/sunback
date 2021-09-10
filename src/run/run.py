import os
import sys

# # Main Command Structure
from time import sleep, time

import numpy as np
class Runner:
    def __init__(self, params):
        self.params = params
        self.wall_1 = "*****************************************************************"
        self.wall_2 = "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_"
    
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
        print(self.wall_2)
        # print(self.params.runner_name)
        print("Starting Batch: {}".format(self.params.batch_name()))
        print(self.wall_2, "\n")
        self.params.set_waves_to_do()
        
        for wave in self.params.waves_to_do:
            self.params.current_wave(wave)

            if len(self.params.fetchers()) > 0:
                sys.stdout.flush()
                print("\n>>>>>>>>>> Fetching Images <<<<<<<<<<\n", flush=True)
                # print(" Redownload Mode: {}\n".format(self.params.redownload_files()))
                for fet in self.params.fetchers():
                    sleep(0.01)
                    fet.fetch(self.params)
                    
                    sleep(0.01)
            
            if len(self.params.processors()) > 0:
                sys.stdout.flush()
                print(">>>>>>>>>> Processing Images <<<<<<<<<<", flush=True)
                # print(" Reprocess Mode: {}\n".format(self.params.reprocess_mode()))
                sys.stdout.flush()
                for proc in self.params.processors():
                    sleep(0.01)
                    proc.process(self.params)
                    sleep(0.01)
                    
            if len(self.params.putters()) > 0:
                sys.stdout.flush()
                print(">>>>>>>>>> Outputting Images or Movies <<<<<<<<<<", flush=True)
                # print(" Redo Imgs: {}".format(self.params.overwrite_pngs()))
                # print(" Redo Videos: {}".format(self.params.write_video()))
                for put in self.params.putters():
                    sleep(0.01)
                    put.put(self.params)
                    sleep(0.01)
            
            
            self.print_end_banner()

    def __process_parallel(self):
        """Use the provided fetcher, executor,
        and putter to do the thing"""
        print(self.wall_2)
        # print(self.params.runner_name)
        print("Starting Batch: {}".format(self.params.batch_name()))
        print(self.wall_2, "\n")

        from joblib import Parallel, delayed
        # the_output = Parallel(n_jobs=-1)(delayed(yourfunction)(k) for k in range(1,10))
        

        if len(self.params.fetchers()) > 0:
            sys.stdout.flush()
            print(">>>>>>>>>> Fetching Images <<<<<<<<<<\n", flush=True)
            # print(" Redownload Mode: {}\n".format(self.params.redownload_files()))
            for fet in self.params.fetchers():
                sleep(0.01)
                fet.fetch(self.params)
                sleep(0.01)
        
        if len(self.params.processors()) > 0:
            sys.stdout.flush()
            print(">>>>>>>>>> Processing Images <<<<<<<<<<", flush=True)
            # print(" Reprocess Mode: {}\n".format(self.params.reprocess_mode()))
            sys.stdout.flush()
            for proc in self.params.processors():
                sleep(0.01)
                proc.process(self.params)
                sleep(0.01)
                
        if len(self.params.putters()) > 0:
            sys.stdout.flush()
            print(">>>>>>>>>> Outputting Images or Movies <<<<<<<<<<", flush=True)
            # print(" Redo Imgs: {}".format(self.params.overwrite_pngs()))
            # print(" Redo Videos: {}".format(self.params.write_video()))
            for put in self.params.putters():
                sleep(0.01)
                put.put(self.params)
                sleep(0.01)
        
        self.print_end_banner()


    ## PRINTING
    def print_header(self):
        print("\n\n", self.wall_1)
        print("\nSunback SDO Image Manipulator \nWritten by C. R. Gilly")
        print("Check out my website: http://gilly.space\n")
        self.start_timestamp = time()
        if self.params.is_debug(): print("DEBUG MODE\n")
        self.print_plan()
        print("\n", self.wall_1, "\n\n")
        # print("Runner basename: ", self.file_name)
   
    def print_plan(self):
        print("Run Name: {}".format(self.params.batch_name()))
        print("Run Type: {}\n".format(self.params.run_type()))
        print(" Here's the Plan:")
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
        print("\n" + self.wall_2)
        self.elapsed = time() - self.start_timestamp
        self.start_timestamp = time()
        minutes = int(np.floor(self.elapsed/60))
        seconds = int(self.elapsed-minutes*60)
        print("Program Complete in {} minutes and {} seconds. {}".format(minutes, seconds, mode_string))
        print(self.wall_2 + "\n")
        
        # for ii in range(4):
        if self.params.stop_after_one():
            print(r"""           '
                          .      '      .
                    .      .     :     .      .
                     '.        ______       .'
                       '  _.-"`      `"-._ '
                        .'                '.
                 `'--. /                    \ .--'`
                      /                      \
                     ;                        ;
                - -- |                        | -- -
                     |     _.                 |
                     ;    /__`A   ,_          ;
                 .-'  \   |= |;._.}{__       /  '-.
                    _.-""-|.' # '. `  `.-"{}<._
                          / 1938  \     \  x   `"
                     ----/         \_.-'|--X----
                     -=_ |         |    |- X.  =_
                    - __ |_________|_.-'|_X-X##
                    jgs `'-._|_|;:;_.-'` '::.  `"-
                     .:;.      .:.   ::.     '::.
                     """)
                
            print("\n")
        