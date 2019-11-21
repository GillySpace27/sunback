
if __name__ == "__main__": main = True
else: main = False

from mpi4py import MPI
import masterslave as ms
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = rank == 0
size = comm.Get_size()
alone = size < 2


def rprint(thing, **kwargs):
	if root:print(thing, **kwargs)


##Imports
if main and not alone: rprint("Importing Modules", end = "...", flush = True)		
		
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pathlib
from functools import partial
from rebin import rebin

from urllib.request import urlretrieve as urlret
from astropy.io import fits
import astropy.table as tb

from PIL import Image
import warnings
warnings.simplefilter("ignore")
from scipy import interpolate as interp

import os
import os.path as path
import sys
import time

##Unchanging Data
pathToFolder = r"C:\Users\chgi7364\Dropbox\Drive\Pictures\SunToday\\"
pathToFits = pathToFolder +'fits\\'

tempPath = path.normpath(path.join(pathToFolder, "temp_Now.jpg"))
tempsPath = pathToFolder + "temps\\"
tempRefPath = tempsPath + "reference\\"


pathlib.Path(pathToFolder).mkdir(parents=True, exist_ok=True)
pathlib.Path(pathToFits).mkdir(parents=True, exist_ok=True)
pathlib.Path(tempsPath).mkdir(parents=True, exist_ok=True)
pathlib.Path(tempRefPath).mkdir(parents=True, exist_ok=True)

responseFile = "aia_filters_prelaunch.txt"
responsePath = path.join(pathToFolder, responseFile)

# rLabels = ['94/131','131/171','171/193','193/211','211/304','304/355']
labels = ['094','131','171','193','211','304','335']

fileEnd = "_Now.jpg"

minutes = 60
hours =  minutes * 60

updateTime = 1 * minutes

##Parallel Stuff
parallel = True
cores = int(len(labels))

loop = True
roll = -1

kill304 = True
if kill304: labels.pop(5)

confidence = True

doMatch = False

binsize = 1

rLabels = []
for ii in np.arange(len(labels)-1):
	rLabels.append("{}/{}".format(labels[ii], labels[ii+1]))
	
if loop: rLabels.append("{}/{}".format(labels[-1], labels[0]))

if main and not alone: rprint('done', flush = True)


def downloadImage(webPath, fullPath, name, doPrint = True):
	"""Download an image and save it to file"""
	try:urlret(webPath, fullPath)
	except: 
		try:urlret(webPath, fullPath)
		except: 
			if doPrint: print(f'{name}: Failed', flush = True); return False
	if doPrint: print(f'{name}: Success', flush=True)
	return True

def getNewTimeOld(wave):
	"""Return the time of the newest available image"""
	webString = "http://sdowww.lmsal.com/sdomedia/SunInTime/{:04}/{:02}/{:02}/{:04}_time.txt"
	now = time.gmtime()
	
	# wave = labels[0]
	fileEnd = "_time.txt"
	fullPath = path.normpath(pathToFits + wave + fileEnd)
	
	webPath = webString.format(now.tm_year,now.tm_mon,now.tm_mday,int(wave))
	
	success = downloadImage(webPath, fullPath, wave, False)	
	
	if not success:
		webPath = webString.format(now.tm_year,now.tm_mon,now.tm_mday-1,int(wave))
		success = downloadImage(webPath, fullPath, wave, False)	
	try:
		with open(fullPath, 'r') as fp:
			lines = fp.readlines()
		line = lines[0].rstrip()
	except:
		with open(fullPath, 'r') as fp:
			lines = fp.readlines()
		line = lines[0].rstrip()		
		
	hour = int(line[-6:-4])
	minute = int(line[-4:-2])
	second = int(line[-2:])
	
	return hour, minute, second

def getNewTime():
	"""Return the time of the newest available image"""
	webPath = "http://jsoc.stanford.edu/data/aia/synoptic/mostrecent/image_times"
	now = time.gmtime()
	
	
	fileEnd = "_time.txt"
	fullPath = path.normpath(pathToFits + "imageTimes.txt")

	success = downloadImage(webPath, fullPath, 'times', False)	

	with open(fullPath, 'r') as fp:
		lines = fp.readlines()
		
		
	line = lines[0].rstrip()
	hour = int(line[-6:-4])
	minute = int(line[-4:-2])
	second = int(line[-2:])

	return hour, minute, second
	
	
	
def getLastTime(wave):
	"""Return the observation time of the currently downloaded file"""
	#Define the file
	fullPath = path.normpath(pathToFits + '{}.fits'.format(wave))
	
	# Get the image out of the file
	hdul = fits.open(fullPath)
	hdul.verify('fix')	

	# Determine the time the image was taken
	fullDate = hdul[0].header['DATE-OBS']
	iHour = int(fullDate[-11:-9])
	iMin = int(fullDate[-8:-6])
	iSec = int(fullDate[-5:-3])	
	
	return iHour, iMin, iSec
	
def newDataSimple():	
	"""Return True if there is new data to be downloaded"""

	#Define the file
	fullPath = path.normpath(pathToFits + '171.fits')
	
	
	statbuf = os.stat(fullPath)
	imgTime = statbuf.st_mtime
	now = time.time()
	
	diff = now - imgTime
	# print(diff, updateTime)
	return diff > updateTime
	
def newData(wave = '171'):
	"""Return True if there is new data to be downloaded"""	
	try:
		iHour, iMin, iSec = getLastTime(wave)
		# print(f"Last Image at {iHour}:{iMin}:{iSec}")
		
		nHour, nMin, nSec = getNewTime()
		# print(f"New Image at {nHour}:{nMin}:{nSec}")
		
		iTime = iHour * hours + iMin * minutes + iSec #in seconds
		nTime = nHour * hours + nMin * minutes + nSec #in seconds
		
		diff = np.abs(nTime - iTime)
		# print(iTime, nTime, diff, updateTime)

		return diff > updateTime
	except: return True
	
	
def runParallel(force = None):	
	"""This function handles calling this script in parallel"""
	
	#Aquire the Flag
	try: go = int(sys.argv[1])
	except: 
		try: go = sys.argv[1]
		except: go = 1
	if force: go = '-force'
		
	#Decide which branch to take
	if not newData(): 
		if str(go) in '-force': 
			rprint("No New Images...Forcing\n")
			go = 1
		elif go == 0: pass
		else:
			rprint("No New Images...Exiting")
			if main: exit()
			else: return
	elif str(go) in '-force': go = 1
	
	#Start up the MPI Branch
	if go == 1 and parallel:
	
		rprint("\nAIA Temperature Reconstruction,\nwritten by Chris Gilbert\n")
	
		print("Starting MPI...", flush = True)
		start = time.time()
		os.system("mpiexec -n {} python {} 0".format(cores, 'aiaTemp_parallel.pyw'))
		print("Parallel Job Complete in {:0.4} seconds\n".format(time.time()-start))
		if main: exit()	

# if main:
	# 
	
	# try:
	# except SystemExit: exit()
	# except:
		# try:runParallel()
		# except SystemExit: exit()
		# except: rprint("It Can't be Done")
	
	

def deltaTest():
	#Load in the Temperature Response Functions
	responseData = loadResponses()
	
	# plt.figure()
	
	myTemps = np.linspace(5.5,7,10)
	for tt in myTemps:
		tempReq = [tt, 6]
		amounts = [100, 0.01]
		
		#Create a synthetic pixel with arbitrary temperatures in it
		ratios = makePixel(tempReq, amounts, responseData)
		
		#Test that pixel
		bestTemp,conf,  matchLevel = statCompare(ratios, responseData)
		
		temps = responseData[0]
		# matchLevel[matchLevel < 0.2] = 0
		plt.plot(temps, matchLevel)
		plt.scatter(tempReq, np.asarray(amounts)/max(amounts))
		plt.scatter(bestTemp,1, marker = 'x')
		# plt.title(f"Best Fit is {bestTemp}, input is {tempReq}")
		# plt.legend(np.round(myTemps,3))
		plt.show()
	
	

def makePixel(tempReq, amounts = None, responseData = None):
	'''Get a single example pixel with given temperatures'''
	#Unpack response data
	temps, respRats, responses = responseData
	
	#Get a list of the indexes of the desired temperatures
	if not isinstance(tempReq, list): tempReq = [tempReq]
	idxs = [find_nearest_idx(temps, req) for req in tempReq]
	useTemps = [find_nearest(temps, req) for req in tempReq]
	
	#Get the responses at those temperatures
	resps = np.asarray([responses[idx] for idx in idxs])
	
	#Set the relative amounts of plasma
	weights = np.ones_like(resps)
	if amounts is not None: 
		assert len(amounts) == len(tempReq)
		for ii, w in enumerate(amounts):
			weights[ii,:] = w
	else: amounts = np.ones_like(tempReq) #np.asarray([1]*len(tempReq))
				
	resps *= weights

	#Sum the responses over temperature
	response = np.sum(resps, axis = 0)
	

	#Find the ratios
	if loop: ratios = response / np.roll(response, roll)
	else: ratios = response[:-1] / response[1:] 
	
	
	#Plot Stuff
	if False:
		fig, [ax0,ax1] = plt.subplots(ncols = 2)
		
		ax0.bar(useTemps, amounts, width=0.1)
		ax0.set_xlabel("Temperature")
		ax0.set_ylabel("Amount of Plasma")
		ax0.set_title("Plasma Composition")
		
		ax1.bar(rLabels, ratios, log = True)
		ax1.set_xlabel("Line Ratios")
		ax1.set_ylabel("Values")
		ax1.set_title("Light Produced")

		for tick in ax1.get_xticklabels():
			tick.set_rotation(45)
			
		# fig.suptitle("Synthetic Observation")
		plt.tight_layout()
		plt.show(True)

	return ratios

	
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
	
def isothermalReconstruction():
	
	#Load in the Temperature Response Functions
	responseData = loadResponses()
	
	#Download the newest images
	# success = downloadAll()
	# success = True
	if newData(): success = downloadAll()
	else: rprint("Rerunning on Old Data"); success = True
	
	if success:
		#Load in the current images
		images, imgTime = loadImages()
		
		#Determine Temperature for each Pixel
		tempArray, conf, matchLevels = tempExtract(images, responseData)
		
		if root:
			#Save the isothermal image to file
			imageSave(tempArray, conf, imgTime)
			
			#Set the desktop background
			if main: updateBackground()
			
			#Save the images by temperature
		if main and doMatch: matchSave(matchLevels, responseData)
			
			
	else: rprint("Exiting Program")
	
	
def updateBackground(fullPath = tempPath):
	"""Update the system background"""
	rprint("Behold!: ", end = '')
	try: 
		ctypes.windll.user32.SystemParametersInfoW(20, 0, fullPath, 0)
		rprint("Background Updated")
	except:rprint("Failed to update background"); raise
	
	
def loadResponses():
	"""Load in the Temperature Response Functions"""
	rprint("Loading Responses", end = "...", flush = True)
	
	#Load the responses from file
	responses = np.loadtxt(responsePath, skiprows = 11, unpack = True)
	temperatures = responses[0]
	responses = np.delete(responses, 0,0).T
	
	#Delete a filter #This is the line that needs to be edited when removing a filter
	if kill304: responses = np.delete(responses, 5,1)
	
	#Interpolate the functions
	newTemperatures = np.arange(min(temperatures), max(temperatures), 0.005)
	newResponses = np.zeros((len(newTemperatures),responses.shape[1]))
	kind = 'cubic'
	for nn in np.arange(responses.shape[1]):
		newResponses[:, nn] = interp.interp1d(temperatures, responses[:,nn], kind = kind)(newTemperatures)
		
	temperatures, rawTemps = newTemperatures, temperatures
	responses, rawResps = newResponses, responses
	
	#Find the ratios of the responses
	if loop: ratios = responses / np.roll(responses, roll, 1)
	else: ratios = responses[:,:-1] / responses[:,1:] 

	
	#Crop the response functions
	lowCut = 5.5
	highCut = 6.8
	
	low = len(temperatures[temperatures < lowCut])
	high = -len(temperatures[temperatures > highCut])
	
	temperatures = temperatures[low:high]
	ratios = ratios[low:high]
	responses = responses[low:high]
	
	
	
	
	# Plot the response functions
	if False:
		plt.figure()
		plt.semilogy(rawTemps, rawResps, 'o-')
		plt.semilogy(temperatures, responses, '.:')
		
		plt.legend(labels)
		plt.title("Response Functions")
		plt.ylim((10**-31, 10**-23.8))
	
	# Plot the ratios of the response functions
	if False:
		plt.figure()
		plt.semilogy(temperatures, ratios, 'o-')
		plt.xlim((4.5,7.5))
		plt.ylim((10**-3,10**3))
		plt.title("Ratios")
		plt.legend(rLabels)
	
	plt.show(True)

	rprint("done")
	return temperatures, ratios, responses

	

	
	
def downloadAll():
	"""Download all the fits images"""
	# webString = "http://sdowww.lmsal.com/sdomedia/SunInTime/{:04}/{:02}/{:02}/f{:04}.fits"
	# webString2 = "http://sdowww.lmsal.com/sdomedia/SunInTime/mostrecent/f{:04}.fits"
	webString3 = "http://jsoc.stanford.edu/data/aia/synoptic/mostrecent/AIAsynoptic{:04}.fits"
	now = time.gmtime()
	fileEnd = '.fits'
	
	rprint("Downloading Images...", end='',flush = True)
	
	if rank < len(labels):
	
		try: wave = labels[rank]
		except: print("Nothing to Download"); return
		# webPath = webString.format(now.tm_year,now.tm_mon,now.tm_mday,int(wave))
		webPath2 = webString3.format(int(wave))
		fullPath = path.normpath(pathToFits + wave + fileEnd)
		success = downloadImage(webPath2, fullPath, wave, False)
	
	else: success = True
		
	comm.barrier()
	gather = comm.gather(success,root=0)
	if root: allsuccess = False if False in gather else True
	else: allsuccess = None
	allsuccess = comm.bcast(allsuccess, root=0)
	
	# rprint(f"All Images Downloaded Successfully: {allsuccess}")
	
	if allsuccess: rprint("Success")
	else: rprint("Failed")
	
	return allsuccess

def createMask(image):

	mask = np.ones_like(image)

	#Nan the Edges
	height1, width1 = mask.shape
	radius = 0.65 * np.sqrt((height1/2)**2 + (width1/2)**2)
	for ii in np.arange(height1):
		for jj in np.arange(width1):
			i,j = ii - height1/2 , jj - width1/2
			if np.sqrt(i*i+j*j) > radius: mask[ii,jj] = np.nan

	return mask
	
def circleMask(mat, rad=0.65):
	
	if mat.shape[0] != mat.shape[1]:
		raise TypeError('Matrix has to be square')
		
	r = int(rad * np.sqrt((mat.shape[0]/2)**2 + (mat.shape[1]/2)**2))
	
	s = mat.shape[0]
	d = np.abs(np.arange(-s/2 + s%2, s/2 + s%2))
	dm = np.sqrt(d[:, np.newaxis]**2 + d[np.newaxis, :]**2)

	return dm > r #np.logical_and(dm >= r-.5, dm < r+.5)		
		
def loadImages():
	"""Load in the current images"""
	rprint("Loading Images", end = "...", flush = True)
	fileEnd = '.fits'
	
	if time.localtime().tm_isdst: offset = time.altzone / hours
	else: offset = time.timezone / hours
	
	xmain, ymain = 2048, 2048
	import copy
	images = []
	for wave in labels:		
		#Define the file
		fullPath = path.normpath(pathToFits + wave + fileEnd)
		
		#Get the image out of the file
		hdul = fits.open(fullPath)
		hdul.verify('fix')
		raw = copy.deepcopy(hdul[0].data)
		# raw = tb.Table.read(hdul[0].data)
		# print(raw)
		image = np.asarray(raw).astype(np.float)
			
			
		# Clean the input #IS THIS ALLOWED??
		
		image *= 1e2
		image[image<0] = 0
		
		image[circleMask(image, 0.7)] = np.nan
		
		
		#Repoint image
		# x0, y0 = hdul[1].header['X0_MP'],hdul[1].header['Y0_MP']
		# xshift, yshift = int(np.round(x0 - xmain)), int(np.round(y0 - ymain))
		# image = np.roll(image, (xshift, yshift), (0,1))

		#Crop for size reasons
		# small = 2
		# images = images[:,::small,::small]
	
		#Rebin the image for speed reasons
		tileSize = binsize
		image = rebin(image, factor = (tileSize,tileSize), func=np.sum)
		
		#Save in list
		images.append(image)
		
		#Plot the Image
		if False:
			# import pdb; pdb.set_trace()
			plt.figure()
			plt.imshow(image, origin = "lower")
			plt.show()
		
		#Determine the time the image was taken
		fullDate = hdul[0].header['DATE-OBS']
		# import pdb; pdb.set_trace()
		hour = int(fullDate[-11:-9])
		minute = int(fullDate[-8:-6])
		imgHour = int(np.mod(hour - offset,12))
		if imgHour == 0: imgHour = 12
		iMonth = int(fullDate[5:7])
		iDay = int(fullDate[8:10])
		imgTime = (iMonth, iDay, imgHour, minute)
		# print("{}: {}/{}, {}:{}".format(wave, *imgTime))
		
	rprint('done')
	return np.asarray(images), imgTime
	
	
def tempExtract(images, responseData):
	"""Determine Temperature for each Pixel"""
	rprint("Finding Temperatures...", flush = True)
	
	
	#Ratio the images
	if loop: ratios = images / np.roll(images, roll, 0)
	else: ratios = images[:-1] / images[1:]
	# 
	# import pdb; pdb.set_trace()
	#Prepare the objects to be analyzed
	tempArray = np.zeros_like(ratios[0])
	
	conf = None
	ratios = np.transpose(ratios, (1,2,0))	
	sComp = partial(statCompare, responseData = responseData)
	sLComp = partial(sLineComp, responseData = responseData)
	
	#Do the loop
	start = time.time()
	
	height, width, nRat = ratios.shape
	matchLevels = np.empty((height, width, len(responseData[0])), dtype=np.float32)
	# print(matchLevels.shape)
	ratList = ratios.tolist()
	radius = 0.65 * np.sqrt((height/2)**2 + (width/2)**2)
	
	if parallel:
		#Run the parallel algorithm
		temporary = ms.poolMPI(ratList, sLComp, True, True)
		# temporary = comm.bcast(temporary, root=0)
		
		if root:
			[tempArray, conf, matchLevels] = zip(*temporary)
			tempArray = np.asarray(tempArray).reshape(height,width)
			matchLevels = np.asarray(matchLevels, dtype=np.float32)
			
		if doMatch:
			rprint(f"Broadcasting {matchLevels.shape}...", end='', flush=True)
			comm.Bcast(matchLevels, root = 0)
			rprint("Success!")
			

	else:
		#Run the serial algorithm 
		if root:
			for ii in np.arange(height):
				for jj in np.arange(width):
					i,j = ii - height/2 , jj - width/2
					if np.sqrt(i*i+j*j) > radius: tempArray[ii,jj] = np.nan
					else: tempArray[ii,jj], _ = sComp(ratios[ii,jj])
					
				if not np.mod(ii,int(height / 50)): rprint("\b\b\b{:02}%".format(int(ii/height * 100)), end = '', flush = True)
			rprint('\b\b\bdone')
	comm.barrier()	
	rprint("    It took {:0.4} seconds!\n".format(time.time() - start))

	return tempArray, conf, matchLevels
	
def sLineComp(ratios, responseData):
	"""Find the temperature for each pixel in a given line"""
	tempArray = np.zeros(len(ratios))
	conf = np.zeros(len(ratios))
	matchLevels = np.zeros((len(ratios),len(responseData[0])))
	
	for ii, rat in enumerate(ratios):
		tempArray[ii], conf[ii], matchLevels[ii, :] = statCompare(rat, responseData)
	return [tempArray, conf, matchLevels]
		

	
def statCompare(ratios, responseData):
	"""Determine the best fit model for a pixel"""
	
	#Make sure we have an array
	ratios = np.asarray(ratios)
	
	#Return Nan if all Nan in input
	if (np.isnan(ratios)).sum() >= len(ratios): return np.nan, np.nan, np.nan
	
	#Return Nan if any Nan in input
	# if not np.isfinite(np.sum(ratios)): return np.nan, np.nan, np.nan
	# print(f"Ratios = {ratios}")
	
	#Unpack response data
	temps, respRats, responses = responseData
	
	#Check for model matching
	matchLevel = compare(ratios, respRats)
	
	#Clean Nans
	# if not np.isfinite(np.sum(matchLevel)): return np.nan, np.nan, np.nan
	
	#Find Best Match
	bestInd = np.argmax(matchLevel)
	bestTemp = temps[bestInd]
	conf = matchLevel[bestInd]
	
	#Plot the matchLevel
	if False:
		print(matchLevel)
		plt.scatter(bestTemp, matchLevel[bestInd])
		plt.plot(temps, matchLevel, label = '1')
		plt.title(f"temp = {bestTemp}")
		plt.legend()
		plt.show()
	
	return bestTemp, conf, matchLevel
		
def compare(A,B):
	"""Compare a single pixel to all the temperature models"""
	# return np.sum(statistic(A, B),axis=1)/len(A)
	length = (~np.isnan(A)).sum()
	return np.nansum(statistic(A, B),axis=1)/length

def statistic(a,b):
	"""Return how well 'a' matches 'b' """
	pow = 4
	toThe = 1
	return (1 - (  (a - b)**pow / (a**pow + b**pow)  ))**toThe

def matchSave(matchLevels, responseData):
	"""Save the matchLevels as individual images"""
	
	rprint("Plotting matchLevels...", end = "   ", flush=True)
	#Unpack response data
	temps, _ , _= responseData	
	
	#Stretch the Images
	minim = np.min(np.nanmin(matchLevels))
	maxim = np.max(np.nanmax(matchLevels))
	
	# matchLevels = (matchLevels - minim) / (maxim - minim)
	mean = np.nanmean(matchLevels)
	std = np.sqrt(np.nanvar(matchLevels))
	
	sig = 1
	
	vmin = mean - (sig * std)
	vmax = mean + (sig * std)
	
	
	
	my_dpi = 120
	height = 1024
	width = height + 0	

	current_cmap = cm.viridis
	current_cmap.set_bad(color='black')
	# current_cmap.set_under("green")
	# current_cmap.set_over("white")		
	fig = plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
	ax = fig.gca()
	
	
	
	
	totalN = len(temps)
	workers = size
	Neach = np.floor(totalN/workers)
	
	startAt = Neach * rank
	endAt = Neach * (rank + 1)
	
	doInds = np.arange(startAt, endAt, dtype=np.int16)
	# print(f"{rank}: {doInds}", flush = True)
	# comm.barrier()
	doTemps = temps[doInds]
	

	for tt, temp in enumerate(doTemps):
		if np.mod(tt,2):continue
		rprint("\b\b\b{:02}%".format(int(tt/len(doTemps) * 100)), end = '', flush = True)

		#Plot the thing

		
		
		# fig.patch.set_facecolor('black')

		
		plotArray = matchLevels[:,:,tt]
		


		ax.imshow(plotArray, cmap = current_cmap, origin = 'lower', interpolation = "none", vmin = vmin, vmax = vmax)
		ax.set_title(f"Temperature is {temp:0.3} = {int(np.round(10**temp,-3)):,}")
		
		# ax = plt.gca()
		# ax.xaxis.set_visible(False)
		# ax.yaxis.set_visible(False)	
		thisPath = path.normpath(path.join(tempsPath, "{:0.2f}.png".format(temp)))
		
		plt.tight_layout()
		fig.savefig(thisPath, facecolor=fig.get_facecolor(), edgecolor='none')
		
	plt.close()
	rprint('\b\b\bdone')
	# print("Range: {:0.4} - {:0.4}".format(minim, maxim))
		# plt.show()
	
	
def clearbar():
	from matplotlib.colors import colorConverter
	import matplotlib as mpl
	# generate the colors for your colormap
	color1 = colorConverter.to_rgba('black')

	# make the colormaps
	cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color1],256)

	cmap2._init() # create the _lut array, with rgba values

	# create your alpha array and fill the colormap with them.
	# here it is progressive, but you can create whathever you want
	
	darkest = 0.9
	lightest = 0
	
	alphas = np.linspace(darkest,lightest, cmap2.N+3)
	cmap2._lut[:,-1] = alphas	
	return cmap2
	
	
def imageSave(tempArray, conf, imgTime):
	"""Save the image to file"""
	
			
	#Convert to millions
	tempArray = 10**tempArray
	tempArray /= 1e6	
	
	#Create the figure
	my_dpi = 120
	height = 2048
	width = height + 200
	
	fig = plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
	fig.patch.set_facecolor('black')
	
	#Define the colormap
	current_cmap = cm.plasma
	current_cmap.set_bad(color='black')
	current_cmap.set_under("green")
	current_cmap.set_over("white")	
	
	mean = np.nanmean(tempArray)
	std = np.sqrt(np.nanvar(tempArray))
	
	vmin = 0.9 #0
	vmax = 2 #3 #3.5
	# if loop:
		# vmin = mean - 2 * std 
		# vmax = mean + 2 * std
	
	
	#Plot
	plt.imshow(tempArray, cmap = current_cmap, origin = 'lower', vmin = vmin, vmax = vmax, interpolation = "none")
	
	ax = plt.gca()
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	
	# plt.title(f"Loop = {loop}: {roll}, kill304 = {kill304}", color = "white")
	
	
	if True: #Make the Colorbar
		ains = inset_axes(ax,
			width="3%",  # width = 10% of parent_bbox width
			height="50%",  # height : 50%
			loc='lower left',
			bbox_to_anchor=(-0.055, 0.32, 1, 1),
			bbox_transform=ax.transAxes,
			borderpad=0,
			)
		
		cb = plt.colorbar(ax = ax, cax=ains, orientation='vertical', extend = 'both') #, ticks=[round(xx*10.0)/10.0 for xx in linspace(0, 1)])
		
		fg_color = "white"
		
		# set colorbar label plus label color
		cb.set_label('        Temperature (MK)\n{:02}/{:02} {:02}:{:02}'.format(*imgTime), color=fg_color, labelpad=-13, y=1.1, rotation=0, size = 18)

		# set colorbar tick color
		cb.ax.yaxis.set_tick_params(color=fg_color)

		# set colorbar edgecolor 
		# cb.outline.set_edgecolor(fg_color)

		# set colorbar ticklabels
		plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color, size = 18)	
	
	
	#Overplot Confidence Layer
	if confidence: ax.imshow(conf, interpolation='none', cmap=clearbar(), origin='lower', vmin=0.3, vmax = 0.8)

	#Show plot and save
	plt.tight_layout()
	fig.savefig(tempPath, facecolor=fig.get_facecolor(), edgecolor='none')
	
	fileLabel = time.strftime('%y%m%d_%H%M')
	
	refPath = path.normpath(path.join(tempRefPath, "temp_{}.png".format(fileLabel)))
	fig.savefig(refPath, facecolor=fig.get_facecolor(), edgecolor='none')
	# plt.show()
	plt.close()

	return
	

if main: 
	# deltaTest()
	runParallel()
	isothermalReconstruction()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
