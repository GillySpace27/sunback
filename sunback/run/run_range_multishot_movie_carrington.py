import os

from sunback.fetcher.FidoFetcher import FidoFetcher
from sunback.fetcher.FidoTimeIntProcessor import FidoTimeIntProcessor
from sunback.fetcher.LocalFetcher import LocalFetcher
from sunback.processor.ImageProcessorCV import ImageProcessorCV
from sunback.processor.QRNProcessor import (
    QRNProcessor,
    QRNSingleShotProcessor,
    QRNradialFiltProcessor,
    QRNpreProcessor,
)
from sunback.processor.RHEProcessor import RHEProcessor

# from src.processor.QRNProcessor import QRNradialFiltProcessor, QRNpreProcessor
from sunback.processor.ScienceProcessor import ScienceProcessor
from sunback.processor.SunPyProcessor import AIA_PREP_Processor
from sunback.processor.ValidationProcessor import ValidationProcessor
from sunback.processor.VideoProcessor import VideoProcessor
from sunback.processor.Processor import Processor

wv = Processor.write_video_in_directory
from sunback.science.parameters import Parameters
import run

# import matplotlib as mpl

# try:
#     mpl.use('qt5agg')
# except ImportError as e:
#     print(e)
import matplotlib.pyplot as plt

plt.ioff()

# wv(r"D:\sunback_images\MultiRange\Liftoff_l_2013_09_28_000020\0304\imgs\png\orig", file_name="0304.avi", orig=True)

# tstart='2014/11/04 01:00:00', tend='2014/11/08 00:00:00',
# tstart='2016/11/04 01:00:00', tend='2016/11/06 00:00:00',
# dostring = "Beautiful 304_l"
all_wavelengths = [
    "0193",
    "0211",
    "0131",
    "0335",
    "0094",
    "0304",
    "0171",
]
do_wavelengths = all_wavelengths  # ['0211']
do_wavelengths = ["0171", "0193", "0211"]
PNG_FRAME_NAME = "rhe"
# wave_to_use = '0211'


def run_range_multishot_movie(
    batch_name="Solar_Cycle3", wave=None, config=None, wave_to_use=None, alpha=0.35
):
    # Set the Parameters
    p = make_params(batch_name, wave, config, wave_to_use)
    p.do_recent(False)
    p.do_prep = True  # Won't do AIA prep upon download of each frame
    p.alpha = alpha
    p.destroy = True
    p.do_parallel = False
    # p.rhe_targets(["compressed_image"])
    # p.init_pool(6)

    # Set the Processes
    p.fetchers(FidoFetcher, rp=True)  # Gets Fits FIDO
    p.processors([RHEProcessor], rp=True)
    p.processors([ImageProcessorCV], rp=True)  # Makes the PNGs from Fits
    p.putters([VideoProcessor], rp=True)  # Makes the PNGs into a Movie

    # p.processors([FidoTimeIntProcessor],    rp=False)   # Integrate several frames for S/N
    # p.processors([AIA_PREP_Processor],      rp=False)   # Do Sunpy Things # This happens in the fetcher now
    # p.qrn_targets(["primary"])
    # p.processors([QRNpreProcessor],            rp=True)  # Applies the RHE Processor
    # p.processors([QRNradialFiltProcessor],     rp=True)  # Applies the RHE Processor

    # # p.putters([ScienceProcessor],             rp=True)  # Makes the PNGs into a Movie

    # Run the Code
    # print(p.do_one())
    run.Runner(p).start()


def make_configs(wave_to_use):
    c17 = {
        "name": "2013_12h",
        "debug": True,
        "do_one": "0171",
        "stop": True,
        "tstart": "2013/01/01 07:00:00",
        "tend": "2014/01/01 07:00:00",
        "cadence_minutes": 12 * 60,
        "fps": 11,
        "exposure_time": 12 * 10,
        "key_fixed_cadence": None,
        "key_fixed_number": 100,
        "time_preset": None,
    }

    c17 = {
        "name": "Solar_Cycle3",
        "debug": True,
        "do_one": "0171",
        "stop": True,
        "tstart": None,
        "tend": None,
        "cadence_minutes": 0,  # -1,  # 25 * 24 * 60,
        "carrington": (2100, 2110, 6),
        "fps": 5,
        "exposure_time": 12 * 10,
        "key_fixed_cadence": None,
        "key_fixed_number": 100,
        "time_preset": None,
    }

    c16 = {
        "name": "Plumes",
        "debug": True,
        "do_one": "0171",
        "stop": True,
        "tstart": "2019/01/01 00:00:01",
        "tend": "2019/02/01 00:00:01",
        "cadence_minutes": 24 * 60,
        "fps": 5,
        "exposure_time": 12 * 10,
        "key_fixed_cadence": None,
        "key_fixed_number": 100,
        "time_preset": None,
    }

    c15 = {
        "name": "Middling",
        "debug": True,
        "do_one": "0171",
        "stop": True,
        "tstart": "2012/01/01 00:00:00",
        "tend": "2012/02/01 00:00:00",
        "cadence_minutes": 24 * 60,
        "fps": 5,
        "exposure_time": 12 * 10,
        "key_fixed_cadence": None,
        "key_fixed_number": 100,
        "time_preset": None,
    }

    c13 = {
        # Five years every other day
        "name": "Long_Test",
        "debug": True,
        "do_one": "0171",
        "stop": True,
        "tstart": "2012/01/01 00:00:00",
        "tend": "2017/01/01 00:00:00",
        "cadence_minutes": 24 * 60 * 7 / 4,
        "fps": 5,
        "exposure_time": 12 * 10,
        "key_fixed_cadence": None,
        "key_fixed_number": 100,
        "time_preset": None,
    }

    c11 = {
        "name": "The_Long_One",
        "debug": True,
        "do_one": "0171",
        "stop": True,
        "tstart": "2017/01/01 00:00:00",
        "tend": "2018/01/01 00:00:00",
        "cadence_minutes": 24 * 60 * 27,
        "fps": 3,
        "exposure_time": 12 * 10,
        "key_fixed_cadence": None,
        "key_fixed_number": 100,
        "time_preset": None,
    }

    c100 = {
        "name": "Single_Search",
        "debug": True,
        "do_one": wave_to_use,
        "stop": True,  # "tend": '2013/09/30 23:59:59',
        "tstart": "2016/01/01 01:00:00",
        "tend": "2017/01/01 00:00:00",
        "cadence_minutes": 24 * 60 * 27,
        "fps": 10,
        "exposure_time": None,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": "l",
    }

    c8 = {
        "name": "Liftoff",
        "debug": True,
        "do_one": wave_to_use,
        "stop": True,  # "tend": '2013/09/30 23:59:59',
        "tstart": "2013/09/28 00:00:10",
        "tend": "2013/09/28 00:00:22",
        "cadence_minutes": 10,
        "fps": 10,
        "exposure_time": 60,
        "key_fixed_cadence": 1,
        "key_fixed_number": None,
        "time_preset": "l",
    }
    c0 = {
        "name": "Test2",
        "debug": True,
        "do_one": wave_to_use,
        "stop": True,
        "tstart": "2022/01/01 00:00:01",
        "tend": "2022/01/03 00:00:00",
        "cadence_minutes": 60,
        "fps": 12,
        "exposure_time": 60,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": None,
    }
    c00 = {
        "name": "Quizzical2",
        "debug": True,
        "do_one": wave_to_use,
        "stop": True,
        "tstart": "2022/04/01 00:00:01",
        "tend": "2022/04/01 03:00:00",
        "cadence_minutes": 60,
        "fps": 1,
        "exposure_time": 24,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": None,
    }
    c1 = {
        "name": "Decadal",
        "debug": True,
        "do_one": wave_to_use,
        "stop": True,
        "tstart": "2011/01/01 00:00:00",
        "tend": "2012/01/01 00:00:00",
        "cadence_minutes": 60 * 24 / 4,
        "fps": 24,
        "exposure_time": 12 * 6,
        "key_fixed_cadence": 8,
        "key_fixed_number": None,
        "time_preset": None,
    }

    c2 = {
        "name": "Gonzalez",
        "debug": True,
        "do_one": wave_to_use,
        "stop": True,
        "tstart": "2017/04/20 11:11:11",
        "tend": "2017/04/20 14:11:11",
        "cadence_minutes": 36 / 60,
        "fps": None,
        "exposure_time": 36,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": None,
    }

    c3 = {
        "name": "Beautiful 171_l",
        "debug": True,
        "do_one": "0171",
        "stop": True,
        "tstart": "2014/11/04 00:00:01",
        "tend": "2014/11/06 00:00:00",
        "cadence_minutes": None,
        "fps": None,
        "exposure_time": None,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": "l",
    }
    c4 = {
        "name": "Beautiful 211",
        "debug": True,
        "do_one": "0211",
        "stop": True,
        "tstart": "2014/11/04 00:00:01",
        "tend": "2014/11/06 00:00:00",
        "cadence_minutes": None,
        "fps": None,
        "exposure_time": None,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": "p",
    }
    c5 = {
        "name": "Pretty 171",
        "debug": True,
        "do_one": "0171",
        "stop": True,
        "tstart": "2015/11/04 00:00:01",
        "tend": "2015/11/06 00:00:00",
        "cadence_minutes": None,
        "fps": None,
        "exposure_time": None,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": "p",
    }
    c6 = {
        "name": "Short 171",
        "debug": True,
        "do_one": "0171",
        "stop": True,
        "tstart": "2015/11/04 00:00:02",
        "tend": "2015/11/05 00:00:00",
        "cadence_minutes": None,
        "fps": None,
        "exposure_time": None,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": "q",
    }
    c7 = {
        "name": "Liftoff 0304",
        "debug": True,
        "do_one": "0304",
        "stop": True,
        "tstart": "2013/09/29 00:00:02",
        "tend": "2013/10/03 00:00:00",
        "cadence_minutes": 10,
        "fps": 16,
        "exposure_time": 60,
        "key_fixed_cadence": 10,
        "key_fixed_number": None,
        "time_preset": "l",
    }

    c9 = {
        "name": "Liftoff 0193",
        "debug": True,
        "do_one": "0193",
        "stop": True,
        "tstart": "2013/09/29 00:00:01",
        "tend": "2013/10/03 00:00:00",
        "cadence_minutes": 10,
        "fps": 32,
        "exposure_time": 60,
        "key_fixed_cadence": 10,
        "key_fixed_number": None,
        "time_preset": "l",
    }
    c10 = {
        "name": "Liftoff 0211",
        "debug": True,
        "do_one": "0211",
        "stop": True,
        "tstart": "2013/09/29 00:00:00",
        "tend": "2013/10/03 00:00:00",
        "cadence_minutes": None,
        "fps": 10,
        "exposure_time": None,
        "key_fixed_cadence": None,
        "key_fixed_number": 100,
        "time_preset": "p",
    }

    c12 = {
        "name": "Recent 0211",
        "debug": True,
        "do_one": "0211",
        "stop": True,
        "tstart": "2021/10/27 00:00:01",
        "tend": "2021/10/31 00:00:00",
        "cadence_minutes": None,
        "fps": 20,
        "exposure_time": None,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": "l2",
    }
    c14 = {
        "name": "Beautiful 304_p",
        "debug": True,
        "do_one": "0193",
        "stop": True,
        "tstart": "2014/11/04 00:00:01",
        "tend": "2014/11/06 00:00:00",
        "cadence_minutes": 5,
        "fps": None,
        "exposure_time": None,
        "key_fixed_cadence": None,
        "key_fixed_number": None,
        "time_preset": "p",
    }

    ConfigDict = {
        c0["name"]: c0,
        c00["name"]: c00,
        c1["name"]: c1,
        c2["name"]: c2,
        c3["name"]: c3,
        c4["name"]: c4,
        c5["name"]: c5,
        c6["name"]: c6,
        c7["name"]: c7,
        c8["name"]: c8,
        c9["name"]: c9,
        c10["name"]: c10,
        c11["name"]: c11,
        c12["name"]: c12,
        c13["name"]: c13,
        c14["name"]: c14,
        c15["name"]: c15,
        c16["name"]: c16,
        c17["name"]: c17,
        c100["name"]: c100,
    }
    return ConfigDict


def make_params(batch_name=None, wave=None, config=None, wave_to_use=None):
    if wave:
        batch_name = batch_name + " " + wave

    # Set the Parameters
    if not config:
        ConfigDict = make_configs(wave_to_use)
        config = ConfigDict[batch_name]

    p = Parameters()
    p.config = config
    p.destroy = False
    # tstart, tend = self.params.set_time_range_duration(tstart, duration_seconds=60):
    if config["tstart"] is None:
        time_string = None
    else:
        time_string = (
            config["tstart"].replace("/", "_").replace(" ", "_").replace(":", "")
        )
    rng = os.path.normpath(
        "MultiRange/{}_{}_{}".format(config["name"], config["time_preset"], time_string)
    )
    p.batch_name(rng)
    p.run_type("Make Movie of Given Time Range, With Time Integration")
    p.do_one(config["do_one"], config["stop"])
    p.is_debug(config["debug"])
    p.do_cat = True
    p.png_frame_name = PNG_FRAME_NAME
    # p.do_recent(True)
    p.currently_local = True
    p.use_drive = "G"

    # Set the Times
    # if not p.load_preset_time_settings(config["time_preset"]):
    p.cadence_minutes(config["cadence_minutes"])
    p.exposure_time_seconds(config["exposure_time"])
    p.frames_per_second(config["fps"])
    p.fixed_cadence_keyframes(config["key_fixed_cadence"])
    p.fixed_number_keyframes(config["key_fixed_number"])
    p.time_period(period=[config["tstart"], config["tend"]])
    p.carrington(config["carrington"])
    # p.compare_fits_frames()

    return p


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    import numpy as np

    for wave_to_use in do_wavelengths:
        #     for alpha in np.linspace(0.25,0.5,20):
        #         run_range_multishot_movie(wave_to_use=wave_to_use, alpha=alpha)
        #         # break
        run_range_multishot_movie(wave_to_use=wave_to_use)


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
#     p.processors([QRNpreProcessor], rp=True)  # Learns the bounds of the dataset for QRN
#     p.processors([QRNradialFiltProcessor], rp=True)  # Applies the QRN Filter
#
#     p.putters([ImageProcessor], rp=True)  # Makes the PNGs from Fits
#     p.putters([VideoProcessor], rp=True)  # Makes the PNGs into a Movie
#
#     # Run the Code
#     run.Runner(p).pointing_start()
