#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sunpy
# import sunpy.map

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy.units as u
import astropy.visualization as av
from astropy.coordinates import SkyCoord
from astropy.io import fits

import xarray as xr #Used to handle netCDF files

#from panel.interact import interact

import hvplot.xarray

#imports for parsing files

import shutil

import sunpy.visualization.colormaps as cm

import netCDF4 as nc
import datetime
import pandas as pd

import os
import sys
import logging
import pathlib
from pathlib import Path
from sys import platform
from datetime import datetime, timedelta
from filelock import Timeout, FileLock

# import SPRINTS API utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from api.sprints_api import api_utils

if __name__ == "__main__":
    # set logging level
    api_utils.set_log(logging.INFO)

    # obtain lock file
    lock_file = os.path.join('/', str(Path.home()), "download_aia.lock")
    lock = FileLock(lock_file)
    try:
        with lock.acquire(timeout=10):
            #str(Path(__file__).resolve().parent.parent)
            config = api_utils.get_config(os.path.dirname('/home/jupyter-bbarnett/sprints/ingest/measurements/config.properties'))
            config_header = 'AIA' 
        
            history_file = config.get(config_header, 'HISTORY_FILE')
            #history_file = '.history/aia.pickle'

            target_dir = config.get(config_header, 'LOCAL_DIR_LINUX') if (platform == "linux") else config.get(config_header, 'LOCAL_DIR_WIN')
            #target_dir = 'C:/Data/Sprints/AIA'

            file_pattern = config.get(config_header, 'FILE_PATTERN')
            #file_pattern = '.*[f|m]1m.*.fits$'

            #put different wavelengths into a list
            #SUVI_wave_lens = ['94', '131', '171', '195', '284', '304']
            aia_wave_lens = ['0094', '0131', '0171', '0193', '0211', '0304', '0335', '1600', '1700']



            wavelen = 0
            # find out today's date
            today = datetime.now()


            # Init the netCDF file
            #fn = '/home/jupyter-bbarnett/shared/data/SUVI_netCDF/solar.nc'

            #create a name for netCDF based off the current date+time
            dt_pattern = f"{today.strftime('AIA%Y%m%d_')}"
            fn = target_dir + '/' + dt_pattern + '.nc'

            #check if the netCDF needs to be created or loaded 
            if os.path.isfile(fn):
                os.remove(fn)
            
            ds = nc.Dataset(fn, 'w', format='NETCDF4')

            #create all the dimensions for the netCDF
            #time = ds.createDimension('time', None)
            wave_len = ds.createDimension('wave_len', None)

            lat = ds.createDimension('lat', 4096)
            lon = ds.createDimension('lon', 4096)

            ##declare their datatypes
            #times = ds.createVariable('time',np.unicode_,('time'))
            wave_lens = ds.createVariable('wave_len', 'f4', ('wave_len',))

            lats = ds.createVariable('lat', 'f4', ('lat',))
            lons = ds.createVariable('lon', 'f4', ('lon',))

            #make space for the actual values of the light
            #value = ds.createVariable('value', 'f4', ('wave_len', 'time', 'lat', 'lon',))
            value = ds.createVariable('value', 'f4', ('wave_len', 'lat', 'lon',))
            value.units = 'Unknown'

            #init and store lats and lons
            lats[:] = np.arange(0, 4096, 1.0)
            lons[:] = np.arange(0, 4096, 1.0)

            #store the aia wave lengths
            wave_lens[:] = aia_wave_lens


            #Fill the netCDF file with the .fits data

            latest_fits = []
            time = 0
            for path, subdirs, files in os.walk(target_dir):
                for name in files:
                    if str(name[len(str(name))-2:]) != 'nc':
                        file = fits.open(target_dir+'/'+name)
                        wl = -1
                        for i in range(len(aia_wave_lens)):
                            if aia_wave_lens[i] == str(name[1:5]):
                                wl = i
                        #print(file[1].data)
                        value[wl, :, :] = file[1].data# * u.arcsec

 
            ds.close()

           
     

    except Timeout:
        logging.warning(f"Another instance of '{__file__}' is running.")


