from processor.Processor import Processor


class SoundProcessor(Processor):
    def __init__(self):
        pass
    
    def process(self):
        raise NotImplementedError()



# class Sonifier:
#     def __init__(self, params, save_path, name, vid_stem, bpm=None, scale=None, frames_per_second=None):
#         self.params = params
#         self.name = name
#         self.scale = scale
#         self.video_name_stem = vid_stem
#         self.first_frame = True
#         self.init_speeds(bpm, frames_per_second)
#         self.init_paths(save_path)
#         self.init_instruments()
#
#     def init_paths(self, save_path):
#         self.save_path = save_path
#         self.midi_path = join(self.save_path, "{}.mid".format(self.name))
#         self.wav_path = join(self.save_path, "{}.wav".format(self.name))
#         self.score_path = join(self.save_path, "{}.score".format(self.name))
#
#     def init_speeds(self, bpm, frames_per_second):
#         self.skip=1
#         if frames_per_second is None:
#             frames_per_second = self.params.frames_per_second()
#         self.frames_per_second = frames_per_second
#         if bpm is None:
#             bpm = self.params.bpm()
#         self.bpm = bpm
#         self.beats_per_second = bpm/60
#         self.seconds_per_beat = 1/self.beats_per_second
#         self.frames_per_beat = self.frames_per_second * self.seconds_per_beat
#
#     def note_length(self, note):
#         """ Takes in a note type (1,2,4,8,16,32) and returns its duration in seconds"""
#         mult = 2 ** -(np.log(note) / np.log(2) - 2)
#         return np.round(self.seconds_per_beat * mult, 6)
#
#     def skip_frames(self, sec):
#         """Returns the number of frames corresponding to a given number of seconds"""
#         return int(np.round(sec*self.frames_per_second, 0))
#
#     def frame_time(self, frames):
#         """returns the number of seconds corresponding to a frame"""
#         return frames/self.frames_per_second
#
#     def frame_on_beat(self, ind, note=None, sec=None, skip=None):
#         if note is not None:
#             sec = self.note_length(note)
#         if sec is not None:
#             skip = self.skip_frames(sec)
#
#         sec = self.frame_time(skip)
#         go = np.mod(ind, skip) == 0
#         # print("Note: {}, Skip {}, ind: {}, sec= {:0.3f}, go = {}".format(note, skip, ind, sec, go))
#
#         if go:
#             return sec
#         else:
#             return False
#
#     def frame_on_any_beat(self, ind):
#         for ii in np.arange(6):
#             if self.frame_on_beat(ind, 2**ii):
#                 return True
#         return False
#
#     def init_instruments(self):
#         """Initialize all the instruments for this score"""
#         self.song = dsp.buffer()
#         self.instruments = []
#         # self.instruments.append(MaxBeeper(self))
#         # self.instruments.append(MaxBeeperSliceLeft(self))
#         # self.instruments.append(MaxBeeperSliceRight(self))
#         self.instruments.append(Segmentor(self))
#
#     def remove_stats(self, data):
#         array = data[:-5, :]
#         bottom = data[-5, :]
#         min = data[-4, :]
#         mid = data[-3, :]
#         max = data[-2, :]
#         top = data[-1, :]
#         return array, (bottom, min, mid, max, top)
#
#     def sonify_frame(self, processed_image_stats, raw_image, file_idx):
#         """Create a phrase for each instrument using this input"""
#         for inst in self.instruments:
#             if self.first_frame:
#                 inst.init_frame(raw_image)
#             inst.sonify_image(file_idx, processed_image_stats, raw_image)
#         self.first_frame=False
#
#     def generate_track(self, path=None):
#         """Write the song to file"""
#         if path is None:
#             path = self.wav_path
#
#         # if not self.params.allow_muxing():
#         self.create(path)
#
#     def create(self, wav):
#         self.instruments[0].create(wav)
#
#     def play(self):
#         """Play the Generated Sound File"""
#         ps(self.wav_path)
#
#     def thread_lock(self):
#         self.instruments[0].thread_lock()
#
# class Instrument:
#     def __init__(self, soni):
#         """Define the way this instrument should sound"""
#         self.soni = soni
#         self.wav_path = soni.wav_path
#         self.frames_per_second = soni.frames_per_second
#         self.song = soni.song
#         self.counter_dict = dict()
#         self.counter=np.zeros(10)
#
#     def play(self):
#         """Play the Generated Sound File"""
#         ps(self.wav_path)
#
#     def init_frame(self, data):
#
#         self.rez = data.shape[0]
#         self.rezX, self.rezY = data.shape
#         centerPt = self.rez / 2
#         xx, yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
#         xc, yc = xx - centerPt, yy - centerPt
#
#         self.extra_rez = 1
#
#         self.sRadius = 400 * self.extra_rez
#         self.tRadius = self.sRadius * 1.28
#         self.bRadius = self.sRadius * 1.01
#         self.radius = np.sqrt(xc * xc + yc * yc) * self.extra_rez
#         self.theta = np.arctan2(xc,yc)
#         self.rez *= self.extra_rez
#         self.on_disk = self.radius <= self.bRadius
#         self.off_limb = self.radius > self.bRadius
#         self.xx, self.yy = np.meshgrid(np.arange(self.rez), np.arange(self.rez))
#
#         # Create arrays sorted by radius
#         self.rad_flat = self.radius.flatten()
#         self.inds = np.argsort(self.rad_flat)
#         self.rad_sorted = self.rad_flat[self.inds]
#         self.dat_flat = data.flatten()
#         self.dat_sorted = self.dat_flat[self.inds]
#
#         self.video_avi = cv2.VideoWriter(self.soni.video_name_stem.format("_son.avi"), cv2.VideoWriter.fourcc("H", "2", "6", "4"), self.frames_per_second,
#                                     (data.shape[0], data.shape[1]))
#
#     def sort_flat(self, data):
#         dat_sorted = data.flatten()[self.inds]
#         return self.rad_sorted, dat_sorted
#
#     def sonify_image(self, frame_ind, processed_image_stats, raw_image):
#         """Generate a phrase from the given input"""
#         proc_im, stats = self.soni.remove_stats(processed_image_stats)
#         self.frame_ind = frame_ind
#         self.voice(frame_ind, proc_im, raw_image, stats)
#
#     def get_mask(self, dat_out, kind=None):
#         if kind is None:
#             kind = self.kind
#         mask = np.full_like(dat_out, True, dtype=bool)
#         lenX, lenY = mask.shape
#         halfX, halfY = int(lenX/2), int(lenY/2)
#
#         if 'l' in kind:
#             mask[halfX:, :] = False
#         if 'r' in kind:
#             mask[:halfX, :] = False
#         if 'u' in kind:
#             mask[:, :halfY] = False
#         if 'd' in kind:
#             mask[:, halfY:] = False
#
#         mask = np.invert(mask)
#         out = dat_out + 0
#         out[mask] = np.nan
#         return mask, out
#
#     def maxChop(self, array, num, wid=75, plotColor=None, plot=False):
#         """Returns the loc and value of the num highest values in an array"""
#         maxes = []
#
#         new_array = array + 0
#         for ii in np.arange(num):
#             maxInd = np.nanargmax(new_array)
#             maxes.append((maxInd, array[maxInd]))
#             low, high = np.max((maxInd - wid, 0)), np.min((maxInd + wid,len(array)))
#             new_array[low:high] = np.nan
#
#         if plot:
#             for loc, val in maxes:
#                 plt.axvline(loc, c=plotColor)
#                 plt.scatter(loc, val, c=plotColor)
#         return maxes
#
#     def parse_kind(self, kind):
#         if type(kind) in [int, float]:
#             return kind
#         if type(kind) not in [str]:
#             raise TypeError
#         if kind.casefold() in 'SINE    '.casefold(): return  0
#         if kind.casefold() in 'SINEIN  '.casefold(): return  17
#         if kind.casefold() in 'SINEOUT '.casefold(): return  18
#         if kind.casefold() in 'COS     '.casefold(): return  1
#         if kind.casefold() in 'TRI     '.casefold(): return  2
#         if kind.casefold() in 'SAW     '.casefold(): return  3
#         if kind.casefold() in 'RSAW    '.casefold(): return  4
#         if kind.casefold() in 'HANN    '.casefold(): return  5
#         if kind.casefold() in 'HANNIN  '.casefold(): return  21
#         if kind.casefold() in 'HANNOUT '.casefold(): return  22
#         if kind.casefold() in 'HAMM    '.casefold(): return  6
#         if kind.casefold() in 'BLACK   '.casefold(): return  7
#         if kind.casefold() in 'BLACKMAN'.casefold(): return  7
#         if kind.casefold() in 'BART    '.casefold(): return  8
#         if kind.casefold() in 'BARTLETT'.casefold(): return  8
#         if kind.casefold() in 'KAISER  '.casefold(): return  9
#         if kind.casefold() in 'SQUARE  '.casefold(): return  10
#         if kind.casefold() in 'RND     '.casefold(): return  11
#         if kind.casefold() in 'LINE    '.casefold(): return  3 # SAW
#         if kind.casefold() in 'PHASOR  '.casefold(): return  3 # SAW
#         if kind.casefold() in 'SINC    '.casefold(): return  23
#         if kind.casefold() in 'GAUSS   '.casefold(): return  24
#         if kind.casefold() in 'GAUSSIN '.casefold(): return  25
#         if kind.casefold() in 'GAUSSOUT'.casefold(): return  26
#         if kind.casefold() in 'PLUCKIN '.casefold(): return  27
#         if kind.casefold() in 'PLUCKOUT'.casefold(): return  28
#         if kind.casefold() in 'LINEAR  '.casefold(): return  12
#         if kind.casefold() in 'TRUNC   '.casefold(): return  13
#         if kind.casefold() in 'HERMITE '.casefold(): return  14
#         if kind.casefold() in 'CONSTANT'.casefold(): return  15
#         if kind.casefold() in 'GOGINS  '.casefold(): return  16
#
#     def parse_adsr(self, adsr, dur):
#         if adsr is None:
#             adsr = [dur / 10, 0, 1, dur / 10]
#         elif type(adsr) in [int, float]:
#             adsr = [dur / adsr, 0, 1, dur / adsr]
#         return [np.round(it,4) for it in adsr]
#
#     def record_note(self, note, note_props=None, frame_ind=None, delay=0):
#         """Record input note into the song"""
#         if note_props is not None:
#             frame_ind, freq, amp, dur, pan, kind, delay, adsr, beat, *mods = note_props
#
#         self.song.dub(note, delay + (frame_ind / self.frames_per_second))
#         if beat not in self.counter_dict:
#             self.counter_dict[beat] = 0
#         self.counter_dict[beat] += 1
#         return note, note_props
#
#     def make_note_props(self, frame_ind=0, freq=440, amp=0.5, dur=1., pan=0.5, kind=0, delay=0, adsr=0, beat=0, *mods):
#         """Concatenate note properties into a list"""
#         return [frame_ind, np.round(freq, 5), np.round(amp, 5), np.round(dur, 6), np.round(pan, 5), int(kind), np.round(delay, 5), adsr, str(beat), *mods]
#
#     def record_osc_note(self, freq=440, amp=0.5, dur=1., pan=0.5, kind=0, delay=0, adsr=None, beat=0, *mods):
#         """Generate a note and record it into the song"""
#         return self.record_note(*self.make_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat, *mods))
#
#     def split_list(self, notes, nGroups=4):
#         nNotes = len(notes)
#         groups = int(nNotes/nGroups)
#         box=[]
#         for ii in np.arange(nGroups):
#             lowInd = ii*groups
#             highInd = np.min(((ii+1)*groups, len(notes)))
#             box.append(notes[lowInd:highInd])
#         return box
#
#     def keep_in_range(self, freq, low, high):
#         while freq >= high:
#             freq /= 2
#         while freq < low:
#             freq *= 2
#         return freq
#
#     def make_osc_note(self, freq=440, amp=0.5, dur=1., pan=0.5, kind=0, delay=0, adsr=None, beat=0, *mods):
#         """Generate a note"""
#         # Input Cleaning
#         amp = 1 if amp > 1 else 0 if amp < 0 else amp
#         pan = 1 if pan > 1 else 0 if pan < 0 else pan
#         kind = self.parse_kind(kind)
#
#         # Tone Generation
#         oscA = Osc(kind, freq=freq, amp=amp)
#         note = oscA.play(dur)
#         note.pan(pan)
#
#         # Effects
#         adsr = self.parse_adsr(adsr, dur)
#         note = note.adsr(*adsr)
#
#         return note, self.make_note_props(self.frame_ind, freq, amp, dur, pan, kind, delay, adsr, beat, *mods)
#
#     def thread_lock(self):
#         for th in self.threads:
#             th.join()
#
# class MaxBeeper(Instrument):
#     def __init__(self, soni):
#         super().__init__(soni)
#
#     def effects(self, note, note_props):
#         frame_ind, freq, amp, dur, pan, kind, delay, *mods = note_props
#
#         note.adsr(
#             a=dsp.rand(0.05 * dur, 0.2 * dur),  # Attack between 50ms and 200ms
#             d=dsp.rand(0.1 * dur, 0.3 * dur),  # Decay between 100ms and 300ms
#             s=dsp.rand(0.2 * dur, 0.6 * dur),  # Sustain between 10% and 50%
#             r=dsp.rand(1 * dur, 2 * dur)  # Release between 1 and 2 seconds*)
#         )
#
#     def voice(self, frame_ind, processed_image, raw_image):
#         if np.mod(frame_ind, self.skip) == 0:
#             return
#         maxX, maxY = np.unravel_index(np.nanargmax(processed_image), processed_image.shape)
#         theMid = np.mean(raw_image)
#         brightest = raw_image[maxX, maxY]
#
#         freq = maxY
#         amp = np.min((1, brightest / theMid / 8))
#         dur = dsp.rand(1, 1.5) * self.skip / self.frames_per_second
#         pan = np.max((0, np.min((1, maxX / 1024))))
#         kind = 0
#         delay = 0
#
#         self.add_note(freq, amp, dur, pan, kind, delay)
#         self.add_note(freq/2, amp/1.5, dur*2, 1-pan, kind+1, delay+0.5)
#
# class MaxBeeperSliceLeft(Instrument):
#     def __init__(self, soni):
#         super().__init__(soni)
#         self.kind='l'
#         self.skip = 10
#
#     def effects(self, note, note_props):
#         frame_ind, freq, amp, dur, pan, kind, delay, *mods = note_props
#
#         note.adsr(
#             a=dsp.rand(0.05 * dur, 0.2 * dur),  # Attack between 50ms and 200ms
#             d=dsp.rand(0.1 * dur, 0.3 * dur),  # Decay between 100ms and 300ms
#             s=dsp.rand(0.5 * dur, 0.9 * dur),  # Sustain between 10% and 50%
#             r=dsp.rand(1 * dur, 2 * dur)  # Release between 1 and 2 seconds*)
#         )
#
#     def voice(self, frame_ind, processed_image, raw_image):
#         if np.mod(frame_ind, self.skip) == 0:
#             return
#         mask, use_box = self.get_mask(processed_image, 'lu')
#
#         maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
#         theMean = np.mean(raw_image)
#         brightest = raw_image[maxX, maxY]
#
#         freq = maxY
#         amp = np.min((1, brightest / theMean / 8))
#         dur = dsp.rand(1, 1.5) * self.skip / self.frames_per_second
#         pan = 0.5*maxX / 1024
#         kind = 2
#         delay = 0
#
#         self.add_note(freq, amp, dur, pan, kind, delay)
#
#         mask, use_box = self.get_mask(processed_image, 'ld')
#         maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
#         theMean = np.mean(raw_image)
#         brightest = raw_image[maxX, maxY]
#
#         freq = maxY
#         amp = np.min((1, brightest / theMean / 8))
#         dur = 3*dsp.rand(1, 1.5) * self.skip / self.frames_per_second
#         pan = 0.5*maxX / 1024
#         kind = 3
#         delay = 0.5
#
#         self.add_note(freq, amp, dur, pan, kind, delay)
#
# class MaxBeeperSliceRight(Instrument):
#     def __init__(self, soni):
#         super().__init__(soni)
#         self.kind='r'
#         self.skip=5
#
#     def effects(self, note, note_props):
#         frame_ind, freq, amp, dur, pan, kind, delay, *mods = note_props
#
#         note.adsr(
#             a=dsp.rand(0.05 * dur, 0.2 * dur),  # Attack between 50ms and 200ms
#             d=dsp.rand(0.1 * dur, 0.3 * dur),  # Decay between 100ms and 300ms
#             s=dsp.rand(0.1 * dur, 0.5 * dur),  # Sustain between 10% and 50%
#             r=dsp.rand(1 * dur, 2 * dur)  # Release between 1 and 2 seconds*)
#         )
#
#     def voice(self, frame_ind, processed_image, raw_image):
#         if np.mod(frame_ind, self.skip) == 0:
#             return
#         mask, use_box = self.get_mask(processed_image, 'ru')
#
#         maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
#         theMean = np.mean(raw_image)
#         brightest = raw_image[maxX, maxY]
#
#         freq = maxY
#         amp = np.min((1, brightest / theMean / 8))
#         dur = dsp.rand(1, 1.5) * self.skip / self.frames_per_second
#         pan = (maxX / 1024)/2 + 0.5
#         kind = 0
#         delay = 0
#
#         self.add_note(freq, amp, dur, pan, kind, delay)
#
#         mask, use_box = self.get_mask(processed_image, 'rd')
#         maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
#         theMean = np.mean(raw_image)
#         brightest = raw_image[maxX, maxY]
#
#         freq = maxY / 2
#         amp = np.min((1, brightest / theMean / 8))
#         dur = 3 * dsp.rand(1, 1.5) * self.skip / self.frames_per_second
#         pan = (maxX / 1024)/2 + 0.5
#         kind = 1
#         delay = 0.5
#
#         self.add_note(freq, amp, dur, pan, kind, delay)
#
# from proglog import TqdmProgressBarLogger
#
#
# class Segmentor(Instrument):
#     def __init__(self, soni):
#         super().__init__(soni)
#         from collections import defaultdict
#         self.strain = defaultdict(lambda :True)
#         self.strain8 = False
#         self.strain4 = False
#         self.ax=None
#         self.tries = 0
#         self.cutoff = 127
#         self.max_tries = 20
#         self.mean_adjust=-60
#         self.imbox = []
#
#         self.chord = tune.next_chord("I")
#
#     def create(self, path):
#         # print(">Writing Sound...", end="")
#         print(self.counter_dict)
#         self.sound_writer(path)
#         self.movie_writer()
#         # print("Success!")
#
#         # Thread(target=self.movie_writer).start()
#         # Thread(target=self.play).start()
#
#     def play_movie(self):
#         bbb.wait()
#         # startfile(self.soni.video_name_stem.format("_son.mp4"))
#
#     def write_wrapper_son_hq(self, videoclip_full_muxed):
#         videoclip_full_muxed.write_videofile(self.soni.video_name_stem.format("_son.mp4"), codec='libx264', bitrate='200M',
#                                              logger=TqdmProgressBarLogger(print_messages=False))
#         bbb.wait()
#     def write_wrapper_son_lq(self, videoclip_full_muxed):
#         bbb.wait()
#         print("Starting")
#         videoclip_full_muxed.write_videofile(self.soni.video_name_stem.format("_son_lq.mp4"), codec='libx264', bitrate='5M',
#                                              logger=TqdmProgressBarLogger(print_messages=False))
#         print("Done")
#     def movie_writer(self):
#
#         cv2.destroyAllWindows()
#         self.video_avi.release()
#
#         videoclip_full = VideoFileClip(self.soni.video_name_stem.format("_son.avi"))
#         videoclip_full_muxed = videoclip_full.set_audio(AudioFileClip(self.wav_path))
#
#         hq_sonFunc = partial(self.write_wrapper_son_hq, videoclip_full_muxed)
#         lq_sonFunc = partial(self.write_wrapper_son_lq, videoclip_full_muxed)
#
#         t1 = Thread(target=hq_sonFunc)
#         t2 = Thread(target=lq_sonFunc)
#         t3 = Thread(target=self.play_movie)
#         t1.start()
#         t2.start()
#         t3.start()
#         self.threads = [t1,t2,t3]
#         # videoclip_full_muxed.write_videofile(self.soni.video_name_stem.format("_son.mp4"), codec='libx264', bitrate='200M',
#         #                                      logger=TqdmProgressBarLogger(print_messages=False))
#         # videoclip_full_muxed.write_videofile(self.soni.video_name_stem.format("_son_lq.mp4"), codec='libx264', bitrate='5M',
#         #                                      logger=TqdmProgressBarLogger(print_messages=False))
#         # for ii in np.arange(3):
#         #     try:
#         #         remove(self.soni.video_name_stem.format("_son.avi"))
#         #     except:
#         #         continue
#         #     break
#
#     def sound_writer(self, path):
#         self.wav_path = path
#         fx.norm(self.song, 1)
#         self.song.write(self.wav_path)
#
#
#     def radial_notes(self, stats):
#         btm, mina, mida, maxa, top = stats
#
#         # rad_sort1, dat_sort = self.sort_flat(raw_image)
#         # rad_sort2, dat_sort_proc = self.sort_flat(processed_image)
#         #
#         # theMaxHere=np.max(maxa)
#         # dat_sort_proc_tall = [theMaxHere*x for x in dat_sort_proc]
#         # plt.figure()
#         # plt.plot(aStat, btm, 'darkred')
#         # plt.plot(aStat, mina, 'r')
#         # plt.plot(aStat, mida, 'g')
#         # plt.plot(aStat, maxa, 'b')
#         # plt.plot(aStat, top, 'darkblue')
#         # plt.scatter(rad_sort1, dat_sort, c='k')
#         # # plt.scatter(rad_sort2, dat_sort_proc_tall, c='lightgrey')
#         # plt.show()
#
#         low = mina/btm
#         med = mida/mina
#         high = maxa/mida
#         vhigh = top/maxa
#
#         doPlot = False
#
#         if doPlot:
#             plt.figure()
#             aStat = np.arange(len(mina))
#             plt.plot(aStat, low, 'g')
#             plt.plot(aStat, med, 'r')
#             plt.plot(aStat, high, 'b')
#             plt.plot(aStat, vhigh, 'y')
#         notes_low   = self.maxChop(low, 2, 150, 'g', doPlot)
#         notes_med   = self.maxChop(med, 2, 125, 'r', doPlot)
#         notes_high  = self.maxChop(high, 4, 75, 'b', doPlot)
#         notes_vhigh = self.maxChop(vhigh, 2, 50,'y', doPlot)
#
#         all_notes = notes_low + notes_med + notes_high + notes_vhigh
#         all_freqs = [note[0] for note in all_notes]
#         sortInds = np.argsort(all_freqs)
#         sorted_notes = [all_notes[ind] for ind in sortInds]
#
#         if doPlot:
#             plt.yscale('log')
#             plt.show()
#         return sorted_notes
#
#
#     def array2uint_proc(self, image):
#         maxa = 1.9 #np.nanmax(image)
#         mina = 0.06 #np.nanmin(image)
#         rescaled = 255 * (image - mina) / (maxa - mina)
#         return np.abs(rescaled).astype('uint8')
#
#     def array2uint(self, image):
#         maxa = np.nanmax(image)
#         mina = np.nanmin(image)
#         rescaled = 255 * (image - mina) / (maxa - mina)
#         return np.abs(rescaled).astype('uint8')
#
#     def grey2clr(self, img_grey):
#         themap = 255 * cm.viridis(img_grey)[:, :, :-1]
#         img_clr = np.abs(themap)
#         # img_clr = img_clr[:, :, [2,1,0]]
#         img_clr = np.flip(img_clr, axis=2).astype('uint8')
#         return img_clr
#
#     def get_regions(self, image, where='both'):
#         import imutils
#
#         # Get Image
#         # img_grey=self.array2uint_proc(image)
#         img_grey=self.array2uint(image)
#         # cv2.cvtColor(image, cv2.)
#         # Remove unimportant regions
#         if where in "on disk":
#             img_grey[self.off_limb] = np.mean(img_grey[self.off_limb])
#         elif where in "off limb":
#             img_grey[self.on_disk] = np.mean(img_grey[self.on_disk])
#
#         cnt_img = img_grey+0
#
#         # Blur to remove spurious hot pixels
#         blurSz = 3
#         blurred = cv2.GaussianBlur(img_grey, (blurSz,blurSz), 0)
#         blurred = self.array2uint(blurred.astype('float32')*blurred.astype('float32'))
#         # self.tries = 0
#         # done=False
#         # cutoff = 50
#         # lowcut = 255
#         # highcut = 0
#         # foundhi = False
#         # foundlo = False
#         # print("")
#         # while True:
#         #
#         #     modded = cv2.threshold(blurred, cutoff, 255, cv2.THRESH_BINARY)[1]
#         #     # Remove the Noise
#         #     ksize = 5
#         #     its = 2
#         #     modded = cv2.morphologyEx(modded, cv2.MORPH_CLOSE, np.ones((ksize,ksize), np.uint8), iterations=its)
#         #     modded = cv2.morphologyEx(modded, cv2.MORPH_OPEN, np.ones((ksize,ksize), np.uint8), iterations=its)
#         #
#         #     # Label the Markers
#         #     ret, markers = cv2.connectedComponents(modded)
#         #     ret -= 1 #Don't count the background as a region
#         #
#         #
#         #     print(foundlo, foundhi, cutoff, ret)
#         #     self.tries += 1
#         #     if self.tries > 100:
#         #         sys.exit()
#         #     # print(ret)
#         #     # import pdb; pdb.set_trace()
#         #     if 8 < ret < 25 and not foundlo:
#         #         lowcut = cutoff
#         #         foundlo = True
#         #     elif 0 < ret < 3 and not foundhi:
#         #         highcut = cutoff
#         #         foundhi = True
#         #     elif not foundlo:
#         #         cutoff += 10
#         #     elif not foundhi:
#         #         # if ret > 50:
#         #         #     cutoff -= 6
#         #         # else:
#         #         cutoff -= 8
#         #
#         #     else:
#         #         break
#         # highcut = 10
#         # lowcut = 200
#         # self.tries = 0
#         # print(highcut, lowcut)
#         # Setup SimpleBlobDetector parameters.
#
#         # Make Detector
#         params = cv2.SimpleBlobDetector_Params()
#         try:
#             params.filterByColor = True
#             params.blobColor = 255
#
#             # Change thresholds
#             params.minThreshold = 100
#             params.maxThreshold = 210
#             params.thresholdStep = 5
#
#             # Filter by Area.
#             params.filterByArea = True
#             params.minArea = 100
#
#             # Filter by Circularity
#             params.filterByCircularity = False
#             params.minCircularity = 0.0
#             params.maxCircularity = 0.5
#
#             # Filter by Convexity
#             params.filterByConvexity = False
#             params.minConvexity = 0.8
#             # params.maxConvexity = 0.99
#
#             # Filter by Inertia
#             params.filterByInertia = False
#             params.maxInertiaRatio = 0.3
#
#             params.minDistBetweenBlobs = 100
#
#             # Create a detector with the parameters
#             ver = (cv2.__version__).split('.')
#             if int(ver[0]) < 3:
#                 detector = cv2.SimpleBlobDetector(params)
#             else:
#                 detector = cv2.SimpleBlobDetector_create(params)
#         except:
#             raise
#
#
#         # Detect the points
#         use = blurred
#         keypoints = detector.detect(use)
#
#         the_notes = []
#         for ii, kpt in enumerate(keypoints):
#             (cX, cY), sz = kpt.pt, kpt.size
#
#             xc, yc = self.xx - cX, self.yy - cY
#             radius = np.sqrt(xc * xc + yc * yc)
#             amp = np.sum(use[radius <= sz/2])
#
#             # amp = np.sum(img_grey[markers==ii])
#
#             # M = cv2.moments(c)
#             # if M["m00"] == 0:
#             #     continue
#             # sz = M["m00"]
#             # # compute the center of the contour
#             # cX = int(M["m10"] / M["m00"])
#             # cY = int(M["m01"] / M["m00"])
#             nt = [np.round(cX, 6), np.round(cY,6), amp, np.round(sz,6)]
#             # print(nt)
#             the_notes.append(nt)
#
#         if len(the_notes) > 0:
#
#             # Sort them
#             the_notes = self.sort_notes(the_notes)
#
#             # Throw out the small ones
#             the_notes = the_notes[:6]
#
#             # for ii, nt in enumerate(the_notes[:-1]):
#             #     if not sz[ii] / sz[0] >= 0.25:
#             #         break
#
#             # Plot
#             im_with_keypoints = cv2.drawKeypoints(use, keypoints, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#             for (cX, cY, amp, sz) in the_notes:
#                 center = (int(cX), int(cY))
#                 # cv2.drawContours(img_clr, [c], -1, (255, 0, 0), 2)
#                 cv2.circle(im_with_keypoints, center, 6, (0, 0, 255), -1)
#                 # (x, y), radius = cv2.minEnclosingCircle(c)
#                 # center = (int(x), int(y))
#                 # radius = int(radius)
#                 cv2.circle(im_with_keypoints, center, int(sz), (0, 255, 0), 2)
#                 # cv2.circle(im_with_keypoints, center, 3, (0,255,0), -1)
#
#
#             if False:
#                 if not self.ax:
#                     fig, self.ax = plt.subplots(1, 1, True, True)
#                     fig.set_size_inches((20, 20))
#                     plt.tight_layout()
#                 # for ii, (x, y, a, s) in enumerate(the_notes):
#                 #     self.ax.scatter(x, y, 50 * a, 'r' if done else 'w')
#                 #     self.ax.set_title("Cutoff {:0.4f}, ret {}".format(self.cutoff, ret))
#                 self.ax.imshow(im_with_keypoints, origin='lower')
#                 plt.pause(0.5)
#                 self.ax.cla()
#                 done = True
#
#             self.video_avi.write(im_with_keypoints)
#
#         return the_notes#, cnts, markers, img_grey
#
#     def sort_notes(self, the_notes, which=3):
#         xx, yy, aa, sz  = zip(*the_notes)
#         stats = xx, yy, aa, sz
#         inds = np.argsort(stats[which])
#         the_notes = [the_notes[i] for i in reversed(inds)]
#         return the_notes
#         # xx, yy, aa, sz = zip(*the_notes)
#         # pass
#
#
#
#
#
#
#
#
#             # cv2.imshow("Keypoints", im_with_keypoints)
#             # cv2.waitKey(0)
#
#             # # Threshold the Image
#             # thresh = cv2.threshold(blurred, self.cutoff, 255, cv2.THRESH_BINARY)[1]
#             # # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, box_size, mean_adjust)
#             # # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, box_size, mean_adjust)
#             #
#             # # Remove the Noise
#             # modded = thresh
#             # ksize = 5
#             # its = 2
#             # modded = cv2.morphologyEx(modded, cv2.MORPH_CLOSE, np.ones((ksize,ksize), np.uint8), iterations=its)
#             # modded = cv2.morphologyEx(modded, cv2.MORPH_OPEN, np.ones((ksize,ksize), np.uint8), iterations=its)
#             #
#             # # Label the Markers
#             # ret, markers = cv2.connectedComponents(modded)
#             # ret -= 1 #Don't count the background as a region
#             #
#             # # Make Contours
#             # cnts = imutils.grab_contours(cv2.findContours(modded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
#             #
#             # the_notes = []
#             # for ii, c in enumerate(cnts):
#             #     M = cv2.moments(c)
#             #     if M["m00"] == 0:
#             #         continue
#             #     amp = np.sum(img_grey[markers==ii])
#             #     sz = M["m00"]
#             #     # compute the center of the contour
#             #     cX = int(M["m10"] / M["m00"])
#             #     cY = int(M["m01"] / M["m00"])
#             #
#             #     nt = [cX, cY, amp, sz]
#             #     the_notes.append(nt)
#             #
#             # # Sort them
#             # xx, yy, aa, sz = zip(*the_notes)
#             # inds = np.argsort(sz)
#             # the_notes = [the_notes[i] for i in reversed(inds)]
#             # cnts = [cnts[i] for i in reversed(inds)]
#             # xx, yy, aa, sz = zip(*the_notes)
#             #
#             # # Throw out the small ones
#             # for ii, nt in enumerate(the_notes[:-1]):
#             #     if not sz[ii] / sz[0] >= 0.25:
#             #         break
#             # the_notes = the_notes[:ii]
#             # cnts = cnts[:ii]
#             # n_cont = len(cnts)
#             #
#             # if False:
#             #     if not self.ax:
#             #         fig, self.ax = plt.subplots(1,1,True,True)
#             #         fig.set_size_inches((20,20))
#             #     for ii, (x,y,a,s) in enumerate(the_notes):
#             #         self.ax.scatter(x,y,50*a, 'r' if done else 'w')
#             #         self.ax.set_title("Cutoff {:0.4f}, ret {}".format(self.cutoff, ret))
#             #     if done:
#             #         self.ax.imshow(img_grey, origin='lower')
#             #         plt.pause(1)
#             #         self.ax.cla()
#             #         self.imbox.append(img_grey)
#             #     else:
#             #         self.ax.imshow(cnt_img, origin='lower')
#             #         plt.pause(0.01)
#             #         self.ax.cla()
#             #
#             #
#             # if n_cont < 2:
#             #     self.cutoff -= 10
#             #     self.mean_adjust +=5
#             #     self.tries += 1
#             # elif n_cont > 5:
#             #     self.cutoff += 8
#             #     self.mean_adjust -=4
#             #     self.tries += 1
#             # else:
#             #     done = True
#             # if self.tries > self.max_tries:
#             #     done = True
#             #
#             # if done:
#             #     # Convert to appropriate type
#             #     # img_clr = self.grey2clr(markers/ret*255)
#             #     img_clr = self.grey2clr(cnt_img)
#             #
#             #     for c, (cX, cY, amp, sz) in zip(cnts, the_notes):
#             #         cv2.drawContours(img_clr, [c], -1, (255, 0, 0), 2)
#             #         cv2.circle(img_clr, (cX, cY), 6, (0, 0, 255), -1)
#             #         (x, y), radius = cv2.minEnclosingCircle(c)
#             #         center = (int(x), int(y))
#             #         radius = int(radius)
#             #         cv2.circle(img_clr, center, radius, (0, 255, 0), 2)
#             #         cv2.circle(img_clr, center, 3, (0,255,0), -1)
#
#
#
#             # self.video_avi.write(img_clr)
#
#             # if False:
#             #     if not self.ax:
#             #         fig, self.ax = plt.subplots(1,1,True,True)
#             #         fig.set_size_inches((20,20))
#             #     # for ii, (x, y, a, s) in enumerate(the_notes):
#             #     #     self.ax.scatter(x, y, 50 * a, 'r' if done else 'w')
#             #     #     self.ax.set_title("Cutoff {:0.4f}, ret {}".format(self.cutoff, ret))
#             #     self.ax.imshow(img_clr, origin='lower')
#             #     plt.pause(0.1)
#             #     self.ax.cla()
#
#             # break
#
#
#         # self.max_tries = 5
#
#         # return the_notes, cnts, markers, img_grey
#
#     def voice(self, frame_ind, processed_image, raw_image, stats):
#
#         # btm, mina, mida, maxa, top = stats
#         #
#         # mask, use_box = self.get_mask(processed_image, 'ru')
#         #
#         # maxX, maxY = np.unravel_index(np.nanargmax(use_box), use_box.shape)
#         # theMean = np.mean(raw_image)
#         # brightest = raw_image[maxX, maxY]
#
#         # the_notes, cnts, markers, img_grey = self.get_regions(processed_image)
#         the_notes = self.get_regions(processed_image)
#
#         if len(the_notes) == 0:
#             return
#
#         note_box = self.split_list(the_notes, 4)
#
#         # sorted_radial_notes = self.radial_notes(stats)
#         # note_box = self.split_list(sorted_radial_notes, 4)
#
#         # Instrument
#         beat = 32
#         active = True
#         notes = the_notes
#         kind = 'cos'  # Waveform
#         adsr = 15 # Envelope
#         reverb = 1.5  # Duration Multiplier
#         jitter = 0.05  # Random offset Seconds
#         variance = 0.05  # Duration Percentage
#         high_chance = 0.8
#         low_chance = 0.1
#         keep_all = True
#         low_hz = 800
#         high_hz = 1200
#         limit_hz = True
#         amp_factor=0.7
#         noteNum = 10
#
#         if self.soni.frame_on_beat(frame_ind, 2):
#             # self.chord = tune.next_chord("I")
#             chrds = ['I', 'iii', 'V']
#             self.chord = chrds[frame_ind%len(chrds)]
#
#         duration = self.soni.frame_on_beat(frame_ind, beat)
#         if duration and active and len(notes)>0:
#             keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
#             self.strain[beat]=False
#             if keep:
#                 self.strain[beat]=True
#                 delay = dsp.rand(-jitter, jitter)
#                 dur = duration*reverb*dsp.rand(1-variance, 1+variance)
#                 # frq_avg = np.mean(all_freq)
#                 # amp_avg = np.mean(all_amp)
#
#                 if not adsr:
#                     a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
#                     d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
#                     s = dsp.rand(0.6, 0.95)  # Sustain between 10% and 50%
#                     r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
#                     adsr = [a, d, s, r]
#
#                 (x0, y0, a0, s0) = notes[0]
#                 if len(notes)>1:
#                     (x1, y1, a1, s1) = notes[1]
#                 else:
#                     (x1, y1, a1, s1) = notes[0]
#
#
#                 for ii, (y, x, a, s) in enumerate(notes):
#                     octRange = self.rezY / 4
#                     yc = y - self.rezY/2
#                     xc = x - self.rezX/2
#                     oct = np.round(y / octRange, 0) + 2
#                     # flr = oct*octRange
#                     # pitch = y % octRange
#                     # freqs = []
#                     # freqs.extend(tune.chord(self.chord, key='C', octave=oct))
#                     # freq = freqs[int(pitch % len(freqs))]
#
#                     # amp = 2*s/(s0+s1) #(s+a)/(s0+a0) #4 * a / (ii + 4) #am/amp_avg * amp_factor #np.min(((am - 1) / 3, 1))
#                     amp = 2*a/(a0+a1)
#                     # import pdb; pdb.set_trace()
#                     r = np.sqrt(xc*xc+yc*yc)
#                     z = (r - self.bRadius)/(self.tRadius - self.bRadius)
#                     t = np.arctan2(y, x) * 180 / 3.14
#                     ct = -np.cos(t)/2 +1.5 # 1 to 2
#                     # if r > self.sRadius*0.8:
#                     hi_freq = z if z > 0 else 0 # 0 to 1
#                     md = 10
#                     mid_freq = (t % md)/md # 0 to 1
#                     low_freq = (y / self.rezY) # 0 to 1
#                     freq = tune.a4*(0.5 + low_freq + 4*mid_freq + 4*hi_freq)/2 # 0-2 # 0-600
#                     # freq = (600* + baseline)
#                     # print(r, hi_freq, mid_freq, low_freq, freq)
#                     # kind = 'cos'
#                     # else:
#                     #     freq = y
#                     #     kind = 'tri'
#                     #     amp /= 2
#
#                     # freq = self.keep_in_range(y, low_hz, high_hz) if limit_hz else y
#
#                     # pan = (x - 100)/(1024-200)
#                     # pan = 0 if x < 512 else 1
#                     # freq = x*2
#                     pan = x/self.rezX
#                     # if pan < 0.5:
#                     #     pan = pan / 6
#                     # else:
#                     #     pan = pan / 6 + (1-1/6)
#                         # continue
#
#                     if ii < noteNum:
#                         self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)
#
#         # Instrument
#         beat = 8
#         active = False
#         # notes = note_box[3]
#         kind = 'cos'  # Waveform
#         adsr = 0 # Envelope
#         reverb = 1.25  # Duration Multiplier
#         jitter = 0.2  # Random offset Seconds
#         variance = 0.15  # Duration Percentage
#         high_chance = 0.8
#         low_chance = 0.1
#         keep_all = True
#         low_hz = 700
#         high_hz = low_hz * 2
#         limit_hz = True
#         amp_factor=0.8
#
#         duration = self.soni.frame_on_beat(frame_ind, beat)
#         if duration and active:
#             keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
#             self.strain[beat]=False
#             if keep:
#                 self.strain[beat]=True
#                 delay = dsp.rand(-jitter, jitter)
#                 dur = duration*reverb*dsp.rand(1-variance, 1+variance)
#
#                 if not adsr:
#                     a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
#                     d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
#                     s = dsp.rand(0.6, 0.95)  # Sustain between 10% and 50%
#                     r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
#                     adsr = [a, d, s, r]
#
#
#                 freq = maxY #self.keep_in_range(maxY, low_hz, high_hz) if limit_hz else frq
#                 amp = amp_factor
#                 pan = maxX/1024
#
#                 self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)
#
#         # Instrument
#         beat = 8
#         active = False
#         notes = note_box[2]
#         kind = 'cos'  # Waveform
#         adsr = 15 # Envelope
#         reverb = 1.75  # Duration Multiplier
#         jitter = 0.1  # Random offset Seconds
#         variance = 0.15  # Duration Percentage
#         high_chance = 0.85
#         low_chance = 0.15
#         keep_all = False
#         low_hz = 500
#         high_hz = low_hz * 2
#         limit_hz = True
#         amp_factor=0.7
#
#         duration = self.soni.frame_on_beat(frame_ind, beat)
#         if duration and active:
#             keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
#             self.strain[beat]=False
#             if keep:
#                 self.strain[beat]=True
#                 delay = dsp.rand(-jitter, jitter)
#                 dur = duration*reverb*dsp.rand(1-variance, 1+variance)
#                 all_freq, all_amp = zip(*notes)
#                 frq_avg = np.mean(all_freq)
#                 amp_avg = np.mean(all_amp)
#
#                 if not adsr:
#                     a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
#                     d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
#                     s = dsp.rand(0.6, 0.95)  # Sustain between 10% and 50%
#                     r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
#                     adsr = [a, d, s, r]
#
#                 for (frq, am) in zip(all_freq, all_amp):
#                     freq = self.keep_in_range(frq, low_hz, high_hz) if limit_hz else frq
#                     amp = am/amp_avg * amp_factor#np.min(((am - 1) / 3, 1))
#                     pan = 1-0.5 * frq/frq_avg
#
#                     self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)
#
#         # Instrument
#         beat = 2
#         active = False
#         notes = note_box[1]
#         kind = 'tri'  # Waveform
#         adsr = 30 # Envelope
#         reverb = 2 # Duration Multiplier
#         jitter = 0.0  # Random offset Seconds
#         variance = 0.05  # Duration Percentage
#         high_chance = 0.95
#         low_chance = 0.8
#         keep_all = False
#         low_hz = 150
#         high_hz = 400
#         limit_hz = True
#         amp_factor = 0.9
#
#         duration = self.soni.frame_on_beat(frame_ind, beat)
#         if duration and active:
#             keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
#             self.strain[beat]=False
#             if keep:
#                 self.strain[beat]=True
#                 delay = dsp.rand(-jitter, jitter)
#                 dur = duration*reverb*dsp.rand(1-variance, 1+variance)
#                 all_freq, all_amp = zip(*notes)
#                 frq_avg = np.mean(all_freq)
#                 amp_avg = np.mean(all_amp)
#
#                 if not adsr:
#                     a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
#                     d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
#                     s = dsp.rand(0.7 * dur, 0.9 * dur)  # Sustain between 10% and 50%
#                     r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
#                     adsr = [a, d, s, r]
#
#                 for (frq, am) in zip(all_freq, all_amp):
#                     freq = self.keep_in_range(frq, low_hz, high_hz) if limit_hz else frq
#                     amp = am/amp_avg * amp_factor#np.min(((am - 1) / 3, 1))
#                     pan = 1-0.5 * frq/frq_avg
#
#                     self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)
#
#         # Instrument
#         beat = 1
#         active = False
#         notes = note_box[0]
#         kind = 'cos'  # Waveform
#         adsr = 30 # Envelope
#         reverb = 2 # Duration Multiplier
#         jitter = 0.0  # Random offset Seconds
#         variance = 0.05  # Duration Percentage
#         high_chance = 0.98
#         low_chance = 0.85
#         keep_all = True
#         low_hz = 30
#         high_hz = 150
#         limit_hz = True
#         amp_factor = 0.7
#
#         duration = self.soni.frame_on_beat(frame_ind, beat)
#         if duration and active:
#             keep = keep_all or dsp.rand(0, 1) < (low_chance if self.strain[beat] else high_chance)
#             self.strain[beat]=False
#             if keep:
#                 self.strain[beat]=True
#                 delay = dsp.rand(-jitter, jitter)
#                 dur = duration*reverb*dsp.rand(1-variance, 1+variance)
#                 all_freq, all_amp = zip(*notes)
#                 frq_avg = np.mean(all_freq)
#                 amp_avg = np.mean(all_amp)
#
#                 if not adsr:
#                     a = dsp.rand(0.025, 0.05)  # Attack between 50ms and 200ms
#                     d = dsp.rand(0.05, 0.1)  # Decay between 100ms and 300ms
#                     s = dsp.rand(0.6, 0.95)  # Sustain between 10% and 50%
#                     r = dsp.rand(1, 2) * (dur - duration)  # Release between 1 and 2 seconds*)
#                     adsr = [a, d, s, r]
#
#                 for (frq, am) in zip(all_freq, all_amp):
#                     freq = self.keep_in_range(frq, low_hz, high_hz) if limit_hz else frq
#                     amp = am/amp_avg * amp_factor#np.min(((am - 1) / 3, 1))
#                     pan = 1-0.5 * frq/frq_avg
#
#                     self.record_osc_note(freq, amp, dur, pan, kind, delay, adsr, beat)
#                     break
# # Available to User
