"""Download near real time AIA files, i.e. any files from yesterday and today
"""

""" Run the script when the minute part of the time is divisible by 6 --> M % 6 = 0 
"""
import os
import sys
import logging
from pathlib import Path
from sys import platform
from datetime import datetime, timedelta
from filelock import Timeout, FileLock
import utils as download_utils

# import SPRINTS API utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from api.sprints_api import api_utils

if __name__ == "__main__":
    # set logging level
    api_utils.set_log(logging.INFO)

    # obtain lock file
    lock_file = os.path.join('/', str(Path.home()), "download_aia.lock")
    lock = FileLock(lock_file)
    try:
        with lock.acquire(timeout=10):



            #retrieve config
            config = api_utils.get_config(str(Path(__file__).resolve().parent.parent))
            config_header = 'AIA' 

            url = 'http://suntoday.lmsal.com/sdomedia/SunInTime/'
        
            history_file = config.get(config_header, 'HISTORY_FILE')
            #history_file = '.history/aia.pickle'

            target_dir = config.get(config_header, 'LOCAL_DIR_LINUX') if (platform == "linux") else config.get(config_header, 'LOCAL_DIR_WIN')
            #target_dir = 'C:/Data/Sprints/AIA'

            file_pattern = config.get(config_header, 'FILE_PATTERN')
            #file_pattern = '.*[f|m]1m.*.fits$'

            #put different wavelengths into a list
            #SUVI_wave_lens = ['94', '131', '171', '195', '284', '304']
            aia_wave_lens = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700']

            # find out today's date
            today = datetime.now()

            # delete the old files
            for file in os.scandir(target_dir):
                os.remove(file.path)

            # download files that might have been updated recently
            for wl in aia_wave_lens:
                # add year and month to URL
                dt_url = url + today.strftime("%Y/%m/%d/")

                # add year,month,day to file name pattern
                index = file_pattern.rfind('.*')

                #dt_pattern = file_pattern[:index+2] + f"s{today.strftime('AIA%Y%m%d_%H%M00_')}" + wl + file_pattern[index:]
                #name of file to download    
                dt_pattern = "f" + wl + ".fits"
                #full_pat = dt_url+dt_pattern
                
                # download
                download_utils.download_http_direct(dt_url, target_dir, history_file, dt_pattern, overwrite=True)
                print('downloaded: ' + dt_url+dt_pattern)

                
    except Timeout:
        logging.warning(f"Another instance of '{__file__}' is running.")
   