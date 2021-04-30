from execute.BackgroundExecutor import BackgroundExecutor
from science.parameters import Parameters
# from run.run import Runner
import run

def run_background(delay=40, debug=False, do_one=False, stop=False):
    p = Parameters()
    p.set_delay_seconds(delay)
    p.do_one(do_one, stop)
    p.is_debug(debug)
    p.executor(BackgroundExecutor(p))
    
    run.Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_background()
