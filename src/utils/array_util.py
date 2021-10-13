from astropy.nddata import block_reduce
from PIL import Image
from os.path import dirname, abspath, join, isdir, split
from os import makedirs, getcwd, listdir
from time import time, localtime, strftime


def reduce_array(frame, center, desired):
    # Reduce the size of the array
    resolution = frame.shape[0]
    if resolution > desired:
        reduce_amount = int(resolution / desired)
        frame = block_reduce(frame, reduce_amount)
        center[0] /= reduce_amount
        center[1] /= reduce_amount
    return frame, center


##  THUMBNAILS
def make_thumbs(rtPath):
    smallPath, bigPath, arcPath = get_thumblinks(rtPath)
    imgDat = Image.open(rtPath)
    imgDat.thumbnail((512, 512))
    imgDat.save(smallPath)
    return smallPath, bigPath, arcPath


def get_thumblinks(rtPath):
    name = split(rtPath)[-1]
    arcPath = "renders/archive/" + "{}_{}".format(int(time()), name)
    smallPath = "renders/thumbs/" + name
    bigPath = 'renders/' + name
    makedirs("renders/thumbs/", exist_ok=True)
    return smallPath, bigPath, arcPath
