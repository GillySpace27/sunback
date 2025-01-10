import os
import sys

# # Main Command Structure
from time import sleep, time, asctime, localtime

import numpy as np
from tqdm import tqdm


class Runner:
    def __init__(self, params):
        self.params = params
        self.wall_1 = "*****************************************************************"
        self.wall_2 = "_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_"

    def start(self, verb=True):
        """Select whether to run or to debug"""
        self.start_timestamp = time()
        if verb: self.print_header()
        self.verb=verb
        if self.params.is_debug():
            self.__debug_mode()
        else:
            self.__run_mode()

    def __debug_mode(self):
        """Run the program in a way that will break"""
        while True:
            self.process()
            if self.params.stop_after_one():
                break

    def __run_mode(self):
        """Run the program in a way that won't break"""

        fail_count = 0
        fail_max = 10

        while True:
            try:
                self.process()
                fail_count -= 1
            except (KeyboardInterrupt, SystemExit):
                print("\n\nOk, I'll Stop. Doot!\n")
                break
            except Exception as error:
                fail_count += 1
                if fail_count < fail_max:
                    out_string = "I failed, but I'm ignoring it. Count: {}/{}\n".format(fail_count, fail_max)
                    print(out_string, error, "\n\n")
                    raise error
                    continue
                else:
                    print("Too Many Failures, I Quit!")
                    sys.exit(1)
            if self.params.stop_after_one():
                break

    def process(self):
        """Use the provided fetcher, executor,
        and putter to do the thing"""
        # if type(self) is SingleRunner:
        #     self.process_single()
        #     return

        print(self.wall_2)
        # print(self.params.runner_name)
        print("Starting Batch: {}".format(self.params.batch_name()))
        print(self.wall_2, "\n")
        self.params.set_waves_to_do()

        for wave in self.params.waves_to_do:
            print("Wavelength = {}".format(self.params.current_wave()), end="")
            self.params.current_wave(wave)

            if len(self.params.fetchers()) > 0:
                sys.stdout.flush()
                print("\n\n>>>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<<")
                print(">>>>>>>>>> Fetching Images <<<<<<<<<<", flush=True)
                print("\r>>>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<<\n")

                # print(" Redownload Mode: {}\n".format(self.params.download_files()))
                for fet, rp in zip(self.params.fetchers(), self.params._fet_rp):
                    fet_instance = fet(params=self.params, rp=rp)
                    fet_instance.tic()
                    fet_instance.fetch()
                    fet_instance.cleanup()
                    # self.params.processors_ran.append(fet_instance)

            if len(self.params.processors()) > 0:
                sys.stdout.flush()
                print("\r\n>>>>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<<<")
                print(">>>>>>>>>> Processing Images <<<<<<<<<<", flush=True)
                print("\r>>>>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<<<\n")

                # print(" Reprocess Mode: {}\n".format(self.params.reprocess_mode()))
                sys.stdout.flush()
                for proc, rp in zip(self.params.processors(), self.params._proc_rp):
                    proc_instance = proc(params=self.params, rp=rp)
                    proc_instance.tic()
                    proc_instance.process()
                    proc_instance.cleanup()

            if len(self.params.putters()) > 0:
                sys.stdout.flush()
                print("\r>>>>>>>>>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<<<<<<<<")
                print(">>>>>>>>>> Outputting Images or Movies <<<<<<<<<<", flush=True)
                print("\r>>>>>>>>>>>>>>>>>>>>>>>>*<<<<<<<<<<<<<<<<<<<<<<<<\n")

                # print(" Redo Imgs: {}".format(self.params.overwrite_pngs()))
                # print(" Redo Videos: {}".format(self.params.write_video()))
                for put, rp in zip(self.params.putters(), self.params._put_rp):
                    put_instance = put(params=self.params, rp=rp)
                    put_instance.tic()
                    try:
                        put_instance.put()
                    except StopIteration:
                        print("StopIteration Raised")
                        break

                put_instance.cleanup()

            self.print_end_banner()

    ## PRINTING
    def print_header(self):
        print("\n\n", self.wall_1, "\n")
        print("          Sunback SDO Image Manipulator ")
        print("              Written by C.R. Gilly")
        print("     Check out my website: http://gilly.space\n")
        self.start_timestamp = time()
        if self.params.is_debug(): print("                    DEBUG MODE\n")
        self.print_plan_start()
        print("\n", self.wall_1, "\n\n")

    def print_plan_start(self):
        end=False
        print("   Here's the Plan:")

        if len(self.params.fetchers()) > 0:
            for fet in self.params.fetchers():
                fet.plan(fet, end=end)

        if len(self.params.processors()) > 0:
            for proc in self.params.processors():
                proc.plan(proc, end=end)

        if len(self.params.putters()) > 0:
            for put in self.params.putters():
                put.plan(put, end=end)

        print("   And Stop After One Loop" if self.params.stop_after_one() else "  And then repeat!")
        # print("\n")
        print("\n Run Name: {}".format(self.params.batch_name()))
        print(" Run Type: {}\n".format(self.params.run_type()))

    def print_plan_end(self):
        end=True
        print("   Summery of Previous Job:")

        for the_proc in self.params.processors_ran:
            the_proc.plan(the_proc, end=end)
        self.params.processors_ran = []

        print("   And Stop After One Loop" if self.params.stop_after_one() else "  And then repeat!")
        # print("\n")
        print(" Run Name: {}".format(self.params.batch_name()))
        print(" Run Type: {}\n".format(self.params.run_type()))

    def print_end_banner(self):
        mode_string = "" if self.params.stop_after_one() else ", Restarting Loop"
        print("\n" + self.wall_2)
        print("Started at {}".format(asctime(localtime(self.start_timestamp))))
        self.elapsed = time() - self.start_timestamp
        self.start_timestamp = time()
        print("Ended  at  {}".format(asctime(localtime(self.start_timestamp))))
        minutes = int(np.floor(self.elapsed / 60))
        seconds = round(self.elapsed - minutes * 60, 3)
        print("  Program Complete in {} minutes and {} seconds. {}".format(minutes, seconds, mode_string))
        self.print_plan_end()
        print(self.wall_2 + "\n")
        delay = self.params.delay_seconds()
        # self.params.multi_pool.close()

        if not self.params.stop_after_one():
            delay = 60*60
            # for i in tqdm(np.arange(delay//10), desc=" ** Waiting for {:0.0f} seconds".format(delay)):
            #     sleep(1*10)
        else:
        # for ii in range(4):
            if self.verb:

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

                sleep(0.5)

                for i in range(5):
                    print("\n  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *")



class SingleRunner(Runner):
    def __init__(self, params=None):
        super().__init__(params)
        self.processor_list = None

    def set_image(self, path=None, image=None):
        """Set up the target image_path"""
        pass

    # def process(self):
    #     self.process_single()

    def process_single(self):
        """Run all of the processors on a single image_path"""
        self.print_start_banner()
        self.assemble_processors()
        self.do_process()
        self.print_end_banner()

    def print_start_banner(self):
        """Display some Pertinent Values"""
        print(self.wall_2)
        print("Starting Job: {}".format(self.params.batch_name()))
        print(self.wall_2, "\n")

    def assemble_processors(self):
        """Concatinate all the types of processor into a single list"""
        self.processor_list = []
        if len(self.params.fetchers()) > 0:
            for fet, rp in zip(self.params.fetchers(), self.params._fet_rp):
                self.processor_list.append(fet(params=self.params, rp=rp))

        if len(self.params.processors()) > 0:
            for proc, rp in zip(self.params.processors(), self.params._proc_rp):
                self.processor_list.append(proc(params=self.params, rp=rp))

        if len(self.params.putters()) > 0:
            for put, rp in zip(self.params.putters(), self.params._put_rp):
                self.processor_list.append(put(params=self.params, rp=rp))

        pass

    def do_process(self):
        """Call each of the processors in order"""

        for proc in self.processor_list:
            # proc.process(image_path=self.params.use_image_path())
            print("\n>>-->>  {}: {}  <<--<<".format(proc.filt_name, proc.description))
            proc.process(params=self.params)
            print('')

    # def image_generator(self):
    #     waves = self.params.set_waves_to_do()
    #     yield None, None
    #     for (image_path, wave) in self.image_generator():
    #         self.params.current_wave(wave)