from parameters import Parameters
from sunback import Sunback


def run(delay=30, debug=False, do_one=False, stop=False, mode="web"):
    p = Parameters()
    p.set_delay_seconds(delay)
    p.do_one(do_one, stop)
    p.run_type(mode)
    p.is_debug(debug)
    
    Sunback(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run()
