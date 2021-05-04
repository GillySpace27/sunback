from execute.AwsExecutor import AwsExecutor
from fetch.WebFetch import WebFetch
from science.parameters import Parameters
import run


def run_aws(delay=100, debug=False, do_one=False, stop=False):
    p = Parameters()
    p.delay_seconds(delay)
    p.do_one(do_one, stop)
    p.is_debug(debug)
    p.executor(AwsExecutor(p))
    p.fetcher(WebFetch())
    
    run.Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_aws(debug=True)
