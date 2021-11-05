import os

from fetcher.FidoFetcher import FidoFetcher
from fetcher.FidoTimeIntProcessor import FidoTimeIntProcessor
from processor.ImageProcessor import ImageProcessor
from processor.ImageProcessorCV import ImageProcessorCV
from processor.ImageProcessorMatplotlib import ImageProcessorMatplotlib
from processor.ImageProcessorPIL import ImageProcessorPIL
from processor.SRNSubProcessors import SRNpreProcessor, SRNradialFiltProcessor
from processor.VideoProcessor import VideoProcessor
from science.parameters import Parameters
import run
import matplotlib.pyplot as plt

plt.ioff()


# tstart='2014/11/04 01:00:00', tend='2014/11/08 00:00:00',
# tstart='2016/11/04 01:00:00', tend='2016/11/06 00:00:00',
dostring = "Liftoff 0211"

def run_range_multishot_movie(batch_name= "Liftoff 0304", wave=None, config=None):
    # Set the Parameters
    p = make_params(batch_name, wave, config)
    p.do_recent(False)
    
    # Set the Processes
    p.fetchers(FidoFetcher, rp=True)                # Gets Fits FIDO
    p.processors([FidoTimeIntProcessor], rp=True)   # Integrate several frames for S/N

    p.processors([SRNpreProcessor],         rp=True)  # Learns the bounds of the dataset for SRN
    p.processors([SRNradialFiltProcessor],  rp=True)  # Applies the SRN Filter

    p.putters([ImageProcessorCV],           rp=True)  # Makes the PNGs from Fits
    p.putters([VideoProcessor],             rp=True)  # Makes the PNGs into a Movie
    
    # Run the Code
    run.Runner(p).start()


def make_params(batch_name=None, wave=None, config=None):
    
    if wave:
        batch_name = batch_name + ' ' + wave
    
    # Set the Parameters
    if not config:
        ConfigDict = make_configs()
        config = ConfigDict[batch_name]
    

        
    p = Parameters()
    p.config = config
    p.destroy = False
    # tstart, tend = self.params.set_time_range_duration(tstart, duration_seconds=60):
    time_string = config["tstart"].replace('/', '_').replace(' ', '_').replace(':', '')
    rng = os.path.normpath("MultiRange\\{}_{}_{}".format(config['name'], config["time_preset"], time_string))
    p.batch_name(rng)
    p.run_type("Make Movie of Given Time Range, With Time Integration")
    p.do_one(config["do_one"], config["stop"])
    p.is_debug(config["debug"])
    p.do_cat = True
    p.png_frame_name = 'SRN'
    p.do_recent(True)
    
    # Set the Times
    # if not p.load_preset_time_settings(config["time_preset"]):
    p.cadence_minutes(config["cadence_minutes"])
    p.exposure_time_seconds(config["exposure_time"])
    p.frames_per_second(config["fps"])
    p.fixed_cadence_keyframes(config["key_fixed_cadence"])
    p.fixed_number_keyframes(config["key_fixed_number"])
    p.time_period(period=[config["tstart"], config["tend"]])
    # p.compare_fits_frames()
    
    return p


def make_configs():
    c0 = {
        "name": "Test",
        "debug": True, "do_one": '0304', "stop": True,
        "tstart": '2013/09/29 00:00:30', "tend": '2013/09/29 03:00:30',
        "cadence_minutes": None, "fps": None, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": None, "time_preset": "p"
    }
    
    c1 = {
        "name": "Beautiful 304",
        "debug": True, "do_one": '0304', "stop": True,
        "tstart": '2014/11/04 00:00:02', "tend": '2014/11/05 00:00:00',
        "cadence_minutes": None, "fps": None, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": None, "time_preset": "p"
    }
    
    c2 = {
        "name": "Beautiful 304_l",
        "debug": True, "do_one": '0304', "stop": True,
        "tstart": '2014/11/04 00:00:01', "tend": '2014/11/06 00:00:00',
        "cadence_minutes": None, "fps": None, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": None, "time_preset": "l"
    }
    
    c3 = {
        "name": "Beautiful 171_l",
        "debug": True, "do_one": '0171', "stop": True,
        "tstart": '2014/11/04 00:00:01', "tend": '2014/11/06 00:00:00',
        "cadence_minutes": None, "fps": None, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": None, "time_preset": "l"
    }
    c4 = {
        "name": "Beautiful 211",
        "debug": True, "do_one": '0211', "stop": True,
        "tstart": '2014/11/04 00:00:01', "tend": '2014/11/06 00:00:00',
        "cadence_minutes": None, "fps": None, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": None, "time_preset": "p"
    }
    c5 = {
        "name": "Pretty 171",
        "debug": True, "do_one": '0171', "stop": True,
        "tstart": '2015/11/04 00:00:01', "tend": '2015/11/06 00:00:00',
        "cadence_minutes": None, "fps": None, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": None, "time_preset": "p"
    }
    c6 = {
        "name": "Short 171",
        "debug": True, "do_one": '0171', "stop": True,
        "tstart": '2015/11/04 00:00:02', "tend": '2015/11/05 00:00:00',
        "cadence_minutes": None, "fps": None, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": None, "time_preset": "q"
    }
    c7 = {
        "name": "Liftoff 0304",
        "debug": True, "do_one": '0304', "stop": True,
        "tstart": '2013/09/29 00:00:02', "tend": '2013/10/03 00:00:00',
        "cadence_minutes": 10, "fps": 16, "exposure_time": 60,
        "key_fixed_cadence": 10, "key_fixed_number": None, "time_preset": "l"
    }
    c8 = {
        "name": "Liftoff 0171",
        "debug": True, "do_one": '0171', "stop": True,
        "tstart": '2013/09/29 00:00:01', "tend": '2013/10/03 00:00:00',
        "cadence_minutes": 10, "fps": 16, "exposure_time": 60,
        "key_fixed_cadence": 10, "key_fixed_number": None, "time_preset": "l"
    }
    c9 = {
        "name": "Liftoff 0193",
        "debug": True, "do_one": '0193', "stop": True,
        "tstart": '2013/09/29 00:00:01', "tend": '2013/10/03 00:00:00',
        "cadence_minutes": 10, "fps": 32, "exposure_time": 60,
        "key_fixed_cadence": 10, "key_fixed_number": None, "time_preset": "l"
    }
    c10 = {
        "name": "Liftoff 0211",
        "debug": True, "do_one": '0211', "stop": True,
        "tstart": '2013/09/29 00:00:00', "tend": '2013/10/03 00:00:00',
        "cadence_minutes": None, "fps": 10, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": 100, "time_preset": "p"
    }

    c11 = {
        "name": "The Long One",
        "debug": True, "do_one": '0171', "stop": True,
        "tstart": '2010/01/01 00:00:00', "tend": '2012/01/01 00:00:00',
        "cadence_minutes": 24*60, "fps": 10, "exposure_time": 36,
        "key_fixed_cadence": None, "key_fixed_number": 100, "time_preset": None
    }
    c12 = {
        "name": "Recent 0211",
        "debug": True, "do_one": '0211', "stop": True,
        "tstart": '2021/10/27 00:00:01', "tend": '2021/10/31 00:00:00',
        "cadence_minutes": None, "fps": 20, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": None, "time_preset": "l2"
    }
    ConfigDict = {
        c0["name"]:   c0,
        c1["name"]:   c1,
        c2["name"]:   c2,
        c3["name"]:   c3,
        c4["name"]:   c4,
        c5["name"]:   c5,
        c6["name"]:   c6,
        c7["name"]:   c7,
        c8["name"]:   c8,
        c9["name"]:   c9,
        c10["name"]: c10,
        c11["name"]: c11,
        c12["name"]: c12,
                  }
    return ConfigDict


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_range_multishot_movie()
    # run_range_multishot_movie(dostring)

































# def run_range_multishot_movie(debug=True, do_one='0304', stop=True,
#                               tstart='2016/11/04 01:00:00', tend='2014/11/06 00:00:00',
#                               cadence_minutes=5, fps=10, exposure_time=24,
#                               key_fixed_cadence=3, key_fixed_number=None, time_preset="p"):
#     # Set the Parameters
#     p = Parameters()
#     # tstart, tend = self.params.set_time_range_duration(tstart, duration_seconds=60):
#     time_string = tstart.replace('/', '_').replace(' ', '_').replace(':', '')
#     rng = "MultiRange\\MRange_{}".format(time_string)
#     p.batch_name(rng)
#     p.run_type("Make Movie of Given Time Range, With Time Integration")
#     p.do_one(do_one, stop)
#     p.is_debug(debug)
#
#     # Set the Times
#     if not p.load_preset_time_settings(time_preset):
#         p.cadence_minutes(cadence_minutes)
#         p.exposure_time_seconds(exposure_time)
#         p.frames_per_second(fps)
#     p.fixed_cadence_keyframes(key_fixed_cadence)
#     p.fixed_number_keyframes(key_fixed_number)
#     p.time_period(period=[tstart, tend])
#
#     # p.compare_fits_frames()
#
#     # Set the Processes
#     # p.fetchers(FidoFetcher)                                     # Gets Fits FIDO
#     # p.processors([FidoTimeIntProcessor])                        # Integrate several frames for S/N
#
#     p.processors([SRNpreProcessor], rp=True)  # Learns the bounds of the dataset for SRN
#     p.processors([SRNradialFiltProcessor], rp=True)  # Applies the SRN Filter
#
#     p.putters([ImageProcessor], rp=True)  # Makes the PNGs from Fits
#     p.putters([VideoProcessor], rp=True)  # Makes the PNGs into a Movie
#
#     # Run the Code
#     run.Runner(p).start()
