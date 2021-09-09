from fetcher.FidoFetcher import FidoFetcher, FidoTimeIntProcessor
from processor.ImageProcessor import ImageProcessor
from processor.SRNProcessor import SRNProcessor, SRNpreProcessor, SRNradialFiltProcessor
from processor.VideoProcessor import VideoProcessor
from science.parameters import Parameters
import run
import matplotlib.pyplot as plt
plt.ioff()


def run_range_multishot_movie(debug=True, do_one=False, stop=True,
                              tstart='2014/11/04 00:00:03', tend='2014/11/07 00:00:00',
                              cadence_minutes=5, fps=10, exposure_time=24,
                              key_fixed_cadence=4, key_fixed_number=None, time_preset="Plaid"):
    
    # Set the Parameters
    p = Parameters()
    # tstart, tend = self.params.make_time_range_duration(tstart, duration_seconds=60):
    time_string = tstart.replace('/', '_').replace(' ', '_').replace(':', '')
    rng = "MultiRange\\MRange_{}".format(time_string)
    p.batch_name(rng)
    
    p.run_type("Make Movie of Given Time Range, With Multishot")
    p.do_one(do_one, stop)
    p.is_debug(debug)
    
    # Set the Times
    p.time_period(period=[tstart, tend])
    
    if not p.load_preset_time_settings(time_preset):
        p.cadence_minutes(cadence_minutes)
        p.exposure_time_seconds(exposure_time)
        p.frames_per_second(fps)
        p.fixed_cadence_keyframes(key_fixed_cadence)
        p.fixed_number_keyframes(key_fixed_number)
    
    # p.compare_fits_frames()
    
    # Set the Processes
    p.fetchers(FidoFetcher(              rp=False))  #rp=False))   # Gets Fits FIDO
    p.processors([FidoTimeIntProcessor(  rp=False)]) #rp=False)])  # Integrate several frames for S/N
    
    # p.processors([SRNProcessor(        rp=False)]) #rp=False)])  # Does SRN on each image individually
    p.processors([SRNpreProcessor(       rp=False)])  # Learns the bounds of the dataset for SRN
    p.processors([SRNradialFiltProcessor(rp=False)]) #, )])  # Applies the SRN Filter
    
    p.putters([ImageProcessor(           rp=True)]) #rp=False)])  # Makes the PNGs from Fits
    p.putters([VideoProcessor(           rp=True)]) #rp=False)])  # Makes the PNGs into a Movie
    
    # Run the Code
    run.Runner(p).start()


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    run_range_multishot_movie()






    # p.putter(AwsPutter(p))        # Uploads the PNGs to AWS
    # p.putter(DesktopPutter(p))        # Runs the Desktop Background Sequence on PNGs
    # p.putters(NullPutter(p))       # Does Nothing with the PNGS


    # p.sonify_limit(False)
    # p.remove_old_images(False)
    # p.make_compressed(True)
    # p.sonify_images(True, True)
    # p.sonify_images(False, False)
    # p.do_171(True)
    # p.do_304(True)

    # # p.bpm(150)
    #


    # p.resolution(2048)

    # Run Flags
    # p.redownload_files(False)
    # p.reprocess_mode(True)  # 'skip'(False), 'redo'(True), 'reset', 'double'
    # p.overwrite_pngs(False)
    # p.remake_norm_curves(False)
    # p.write_video(True)
    # p.delete_old(True)






























































