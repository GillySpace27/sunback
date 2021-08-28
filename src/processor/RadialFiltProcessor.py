from time import time
from processor.Processor import Processor
from science.modify import Modify

set_local_background = False

# Initialization
last_time = time()
start_time = last_time

default_sleep = 30


class RadialFiltProcessor(Processor):
    in_name = 'primary'
    out_name = 'SRN'
    filt_name = '  Radial Filter'
    do_function = Modify
    do_png = False
    description = "Filter the Images Radially with SRN"


