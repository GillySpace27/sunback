

print("\nAIA Temperature Reconstruction,\nwritten by Chris Gilbert\n")

print("Importing Modules", end = "...", flush = True)

#Imports
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import time
import pathlib
from functools import partial

from urllib.request import urlretrieve as urlret
from astropy.io import fits

import os.path as path

from PIL import Image
import warnings
warnings.simplefilter("ignore")

#Unchanging Data
pathToFolder = r"C:\Users\chgi7364\Dropbox\Drive\Pictures\SunToday\\"
pathToFits = pathToFolder +'fits\\'

tempPath = path.join(pathToFolder, "temp_Now.jpg")

pathlib.Path(pathToFolder).mkdir(parents=True, exist_ok=True)
pathlib.Path(pathToFits).mkdir(parents=True, exist_ok=True)

responseFile = "aia_filters_prelaunch.txt"
responsePath = path.join(pathToFolder, responseFile)

labels = ['094','131','171','193','211','304','335'] #[
rLabels = ['94/131','131/171','171/193','193/211','211/304','304/355'] #[

fileEnd = "_Now.jpg"

minutes = 60
hours =  minutes * 60

print('done', flush = True)

#Function Definitions

def isothermalReconstruction():
	
	#Load in the Temperature Response Functions
	responseData = loadResponses()
	
	#Download the newest images
	downloadAll()
	
	#Load in the current images
	images, imgTime = loadImages()
	
	#Determine Temperature for each Pixel
	tempArray = tempExtract(images, responseData)
	
	#Save the image to file
	imageSave(tempArray, imgTime)
	
	#Set the desktop background
	updateBackground()
	
	
def updateBackground(fullPath = tempPath):
	"""Update the system background"""
	try: 
		ctypes.windll.user32.SystemParametersInfoW(20, 0, fullPath, 0)
		print("Background Updated")
	except:print("Failed to update background"); raise
	
	
def loadResponses():
	"""Load in the Temperature Response Functions"""
	print("Loading Responses", end = "...", flush = True)
	
	#Load the responses from file
	responses = np.loadtxt(responsePath, skiprows = 11, unpack = True)
	temperatures = responses[0]
	responses = np.delete(responses, 0,0).T
	
	#Find the ratios of the responses
	ratios = responses[:,:-1] / responses[:,1:]
	
	#Crop the response functions
	lowCut = 5
	highCut = 7.5
	
	low = len(temperatures[temperatures < lowCut])
	high = -len(temperatures[temperatures > highCut])
	
	temperatures = temperatures[low:high]
	ratios = ratios[low:high]
	
	# Plot the raw response functions
	# plt.figure()
	# plt.semilogy(temperatures, responses)
	# plt.legend(labels)
	# plt.title("Response Functions")
	# plt.ylim((10**-31, 10**-23.8))
	
	# Plot the ratios of the response functions
	# plt.figure()
	# plt.semilogy(temperatures, ratios)
	# plt.xlim((4.5,7.5))
	# plt.ylim((10**-3,10**3))
	# plt.title("Ratios")
	# plt.legend(rLabels)
	
	# plt.show()
	
	print("done")
	return temperatures, ratios

	
def downloadImage(webPath, fullPath):
	"""Download an image and save it to file"""
	print("Downloading Image...", end = '', flush = True)
	try:urlret(webPath, fullPath)
	except: 
		try:urlret(webPath, fullPath)
		except: print('Failed'); raise
	print("Success")
	return True
	
	
def downloadAll():
	"""Download all the fits images"""
	webString = "http://sdowww.lmsal.com/sdomedia/SunInTime/{:04}/{:02}/{:02}/f{:04}.fits"	
	now = time.gmtime()
	fileEnd = '.fits'
	
	for wave in labels:
		webPath = webString.format(now.tm_year,now.tm_mon,now.tm_mday,int(wave))
		fullPath = path.normpath(pathToFits + wave + fileEnd)
		downloadImage(webPath, fullPath)
	return
	
	
def loadImages():
	"""Load in the current images"""
	print("Loading Images", end = "...", flush = True)
	fileEnd = '.fits'
	images = []

	if time.localtime().tm_isdst: offset = time.altzone / hours
	else: offset = time.timezone / hours
	
	for wave in labels:		
		#Define the file
		fullPath = path.normpath(pathToFits + wave + fileEnd)
		
		#Get the image out of the file
		hdul = fits.open(fullPath)
		hdul.verify('fix')
		image = np.asarray(hdul[1].data).astype(np.float)
				
		#Clean the input #IS THIS ALLOWED??
		image[image<1] = 1
		
		#Save in list
		images.append(image)
		
		#Plot the Image
		# import pdb; pdb.set_trace()
		# plt.imshow(image, origin = "lower")
		# plt.show()
		
		
		#Extract the time it was taken
		fullDate = hdul[1].header['DATE-OBS']
		hour = int(fullDate[-11:-9])
		minute = int(fullDate[-8:-6])
		imgHour = int(np.mod(hour - offset,12))
		if imgHour == 0: imgHour = 12
		imgTime = (imgHour, minute)
		
	print('done')
	return np.asarray(images), imgTime
	
	
def tempExtract(images, responseData):
	"""Determine Temperature for each Pixel"""
	print("Finding Temperatures...", flush = True, end = "   ")
	
	#Crop for size reasons
	small = 4
	images = images[:,::small,::small]
	
	#Ratio the images
	ratios = images[:-1] / images[1:]
	
	#Prepare the objects to be analyzed
	tempArray = np.zeros_like(ratios[0])
	ratios = np.transpose(ratios, (1,2,0))	
	sComp = partial(statCompare, responseData = responseData)
	
	#Do the loop
	start = time.time()
	
	height, width, _ = ratios.shape
	radius = 0.65 * np.sqrt((height/2)**2 + (width/2)**2)
	
	for ii in np.arange(height):
		for jj in np.arange(width):
			i,j = ii - height/2 , jj - width/2
			if np.sqrt(i*i+j*j) > radius: tempArray[ii,jj] = np.nan
			else: tempArray[ii,jj] = sComp(ratios[ii,jj])
			
		#Print percentages	
		if not np.mod(ii,int(height / 50)): print("\b\b\b{:02}%".format(int(ii/height * 100)), end = '', flush = True)
	
	print('\b\b\bdone')
	print("    It took {:0.4} seconds!\n".format(time.time() - start))
	

	return tempArray
	
def statCompare(ratios, responseData):
	"""Determine the best fit model for a pixel"""
	
	#Return Nan if any Nan in input
	if not np.isfinite(np.sum(ratios)): return np.nan
	# print(f"Ratios = {ratios}")
	
	#Unpack response data
	temps, respRats = responseData
	
	#Check for model matching
	matchLevel = compare(ratios, respRats)
	
	#Clean Nans
	if not np.isfinite(np.sum(matchLevel)): return np.nan
	
	#Find Best Match
	bestInd = np.argmax(matchLevel)
	bestTemp = temps[bestInd]
	
	#Plot the matchLevel
	# print(matchLevel)
	# plt.scatter(bestTemp, matchLevel[bestInd])
	# plt.plot(temps, matchLevel, label = '1')
	# plt.plot(temps, matchLevel2, label = '2')
	# plt.title(f"temp = {bestTemp}")
	# plt.legend()
	# plt.show()
	
	return bestTemp
		
def compare(A,B):
	"""Compare a single pixel to all the temperature models"""
	return np.sum(statistic(A, B),axis=1)/len(A)

def statistic(a,b):
	"""Return how well 'a' matches 'b' """
	return 1 - (  (a - b)**2 / (a**2 + b**2)  )

	
def imageSave(tempArray, imgTime):
	"""Save the image to file"""
	
	#Plot the thing
	print("Behold!\n")
	
	my_dpi = 120
	height = 2048
	width = height + 200
	
	fig = plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
	fig.patch.set_facecolor('black')
	
	current_cmap = cm.plasma
	current_cmap.set_bad(color='black')
	current_cmap.set_under("blue")
	current_cmap.set_over("red")	
	
	mean = np.nanmean(tempArray)
	std = np.sqrt(np.nanvar(tempArray))
	
	# print(mean, std)
	
	vmin = mean - 4 * std 
	vmax = mean + 2.25 * std
	
	plt.imshow(tempArray, cmap = current_cmap, origin = 'lower', vmin = vmin, vmax = vmax)
	
	ax = plt.gca()
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	
	
	
	#Make the Colorbar
	
	if True:
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
		cb.set_label('Temperature\n{:02}:{:02}'.format(*imgTime), color=fg_color, labelpad=-13, y=1.1, rotation=0, size = 18)

		# set colorbar tick color
		cb.ax.yaxis.set_tick_params(color=fg_color)

		# set colorbar edgecolor 
		# cb.outline.set_edgecolor(fg_color)

		# set colorbar ticklabels
		plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color, size = 18)	
	
	
	plt.tight_layout()
	fig.savefig(tempPath, facecolor=fig.get_facecolor(), edgecolor='none')
	# plt.show()
	plt.close()

	return
	

isothermalReconstruction()
