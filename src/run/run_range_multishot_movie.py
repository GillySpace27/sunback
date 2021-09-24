from fetcher.FidoFetcher import FidoFetcher
from fetcher.FidoTimeIntProcessor import FidoTimeIntProcessor
from processor.ImageProcessor import ImageProcessor
from processor.SRNSubProcessors import SRNpreProcessor, SRNradialFiltProcessor
from processor.VideoProcessor import VideoProcessor
from science.parameters import Parameters
import run
import matplotlib.pyplot as plt

plt.ioff()


# tstart='2014/11/04 01:00:00', tend='2014/11/08 00:00:00',
# tstart='2016/11/04 01:00:00', tend='2016/11/06 00:00:00',


def run_range_multishot_movie(config_name=0, config=None):
    # Set the Parameters
    p = make_params(config, config_name)
    
    # Set the Processes
    p.fetchers(FidoFetcher, rp=True)                                     # Gets Fits FIDO
    p.processors([FidoTimeIntProcessor], rp=True)                        # Integrate several frames for S/N
    
    # p.processors([SRNpreProcessor], rp=True)  # Learns the bounds of the dataset for SRN
    # p.processors([SRNradialFiltProcessor], rp=True)  # Applies the SRN Filter

    # p.putters([ImageProcessor], rp=True)  # Makes the PNGs from Fits
    # p.putters([VideoProcessor], rp=True)  # Makes the PNGs into a Movie
    
    # Run the Code
    run.Runner(p).start()


def make_params(config=None, config_name=0):
    # Set the Parameters
    if config is None:
        ConfigDict = make_configs()
        config = ConfigDict[config_name]
    
    p = Parameters()
    # tstart, tend = self.params.set_time_range_duration(tstart, duration_seconds=60):
    time_string = config["tstart"].replace('/', '_').replace(' ', '_').replace(':', '')
    rng = "MultiRange\\MRange_{}".format(time_string)
    p.batch_name(rng)
    p.run_type("Make Movie of Given Time Range, With Time Integration")
    p.do_one(config["do_one"], config["stop"])
    p.is_debug(config["debug"])
    
    # Set the Times
    if not p.load_preset_time_settings(config["time_preset"]):
        p.cadence_minutes(config["cadence_minutes"])
        p.exposure_time_seconds(config["exposure_time"])
        p.frames_per_second(config["fps"])
    p.fixed_cadence_keyframes(config["key_fixed_cadence"])
    p.fixed_number_keyframes(config["key_fixed_number"])
    p.time_period(period=[config["tstart"], config["tend"]])
    # p.compare_fits_frames()
    
    return p


def make_configs():
    c1 = {
        "name": "Beautiful 304",
        "debug": True, "do_one": '0304', "stop": True,
        "tstart": '2014/11/04 00:00:00', "tend": '2014/11/06 00:00:00',
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
    c3 = {
        "name": "Beautiful 211",
        "debug": True, "do_one": '0211', "stop": True,
        "tstart": '2014/11/04 00:00:01', "tend": '2014/11/06 00:00:00',
        "cadence_minutes": None, "fps": None, "exposure_time": None,
        "key_fixed_cadence": None, "key_fixed_number": None, "time_preset": "p"
    }

    ConfigDict = {c1["name"]: c1,
                  c2["name"]: c2,
                  c3["name"]: c3}
    return ConfigDict


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_range_multishot_movie("Beautiful 211")

































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
