from os.path import dirname, abspath, join, isdir
from os import makedirs, getcwd


def discover_best_data_directory():
    """Determine where to store the images"""
    # TODO find a good directory
    subdirectory_name = "sunback_images/test"
    if __file__ in globals():
        ddd = dirname(abspath(__file__))
    else:
        ddd = abspath(getcwd())
    
    while "dropbox".casefold() in ddd.casefold():
        ddd = abspath(join(ddd, ".."))
    
    directory = join(ddd, subdirectory_name)
    if not isdir(directory):
        makedirs(directory)
    return directory
