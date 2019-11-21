import ctypes
import time

from urllib.request import urlretrieve as urlret
from random import shuffle
from os.path import normpath

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

import os
import pytesseract as tes
from numpy import mod
from numpy import floor
 
seconds = 1
minutes = 60
hours =  minutes * 60

pathToFile = r"C:\Users\chgi7364\Dropbox\Drive\Pictures\SunToday\\"
webString = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_{}.jpg"
wavelengths = ['0171','0193','0211','0304','0131','0335','0094','HMIBC','HMIIF']
wavelengths.sort()


# refreshPicsTime = 16 * minutes
# picChangeTime = refreshPicsTime / len(wavelengths)

picChangeTime = 0.25 # seconds  #* minutes

#Function Definitions

def downloadImage(webPath, fullPath):
	"""Download an image and save it to file"""
	print("Downloading Image...", end = '')
	try:urlret(webPath, fullPath)
	except: 
		try:urlret(webPath, fullPath)
		except: print('Failed'); return False
	print("Success")
	return True

def updateBackground(fullPath):
	"""Update the system background"""
	try: 
		drawCurrentTime(fullPath)
		ctypes.windll.user32.SystemParametersInfoW(20, 0, fullPath, 0)
		# print("Background Updated\n")
	except:pass
	
def saveReference(pathToFile, wave):
	"""Grab a single reference image and save"""
	try:
		print("Saving Reference Image...", end ='')
		now = time.strftime("%y-%j-%H-%M")
		webpath = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_{}.jpg".format(wave)
		fullpath = normpath(pathToFile + "reference\\171_reference_{}.jpg".format(now))
		urlret(webpath, fullpath)
		print("Success")
	except: print("Failed")
	
	
def readTime(fullPath, wave):
	img = Image.open(fullPath)
	if time.localtime().tm_isdst: offset = time.altzone / hours
	else: offset = time.timezone / hours
	
	try:
		cropped = img.crop((0,1950, 1024, 2048))
		# cropped.show()
		results = tes.image_to_string(cropped)
		
		if wave[0]=='H': #HMI Data
			imgTime = results[-6:]
			imgHour = int(imgTime[:2]) 
			imgMin = int(imgTime[2:4])
		
		else: #AIA Data
			imgTime = results[-11:-6]
			imgHour = int(imgTime[:2])
			imgMin = int(imgTime[-2:])
			
		imgHour = int(mod(imgHour - offset,12))
		pre = ''
	except:
		imgHour = mod(time.localtime().tm_hour,12)
		imgMin = time.localtime().tm_min
		pre = 'x'

	if imgHour == 0: imgHour = 12
	return imgHour, imgMin, pre
	
def imageMod(fullPath, wave):
	try: #Modify the image
		print('Modifying Image...', end='')
		#Open the image for modification
		img = Image.open(fullPath)
		img_raw = img
		font = ImageFont.truetype(normpath("C:\Windows\Fonts\Arial.ttf"), 42)
		
		#Shrink the HMI images to be the same size
		if wave[0] == 'H':
		
			smallSize = 1725
			old_img = img.resize((smallSize,smallSize))
			old_size = old_img.size

			new_size = (2048, 2048)
			new_im = Image.new("RGB", new_size)
			
			x = int((new_size[0]-old_size[0])/2)
			y = int((new_size[1]-old_size[1])/2)

			new_im.paste(old_img, (x, y))
			img = new_im
						
		#Read the time and reprint it 
		imgHour, imgMin, pre = readTime(fullPath, wave)
		#Draw on the image and save
		draw = ImageDraw.Draw(img)
		
		towrite = wave[1:] if wave[0] =='0' else wave
		
		draw.text((1510, 300),towrite,(200,200,200),font=font) 
		draw.text((450, 300),"{:0>2}:{:0>2}{}".format(imgHour, imgMin, pre),(200,200,200),font=font)
		img.save(fullPath)
		print("Success")
	except: print("Failed")
	return


def drawCurrentTime(fullPath):
	img = Image.open(fullPath)
	draw = ImageDraw.Draw(img)
	font = ImageFont.truetype(normpath("C:\Windows\Fonts\Arial.ttf"), 42)
	draw.rectangle([(450,150),(560,200)], fill=(0,0,0))
	draw.text((450, 150),time.strftime("%I:%M"),(200,200,200),font=font)
	img.save(fullPath)	
	
##The Main Loop

print("\nLive SDO Background Updater \nWritten by Gilly\n") 
# fileEnd = "_Now.jpg"

# while True:

	# for wave in wavelengths:
		# try:
			# # Define the Image
			# print("Image: {}".format(wave))
			# webPath = webString.format(wave)
			# full_path = normpath(pathToFile + wave + fileEnd)
			
			# # Download the Image
			# # downloadImage(webPath, full_path)
			
			# # Modify the Image
			# # imageMod(full_path, wave)
			
			# # Update the Background
			# updateBackground(full_path)
			
			# # Wait for a bit
			# time.sleep(picChangeTime)
		# except (KeyboardInterrupt, SystemExit):print("Fine, I'll Stop.\n"); raise
		# except: print("I failed")
	# # saveReference(pathToFile, '0211')

	

import pathlib
import numpy as np

from os import listdir
import sys
	
from shutil import copyfile

loopTime = 45.0 * seconds
frameTime = 2.0 * seconds	
maxFrameCount = 10



#####

while True:
	#Each loop will grab the next clip
	
	#For each wavelength
	for wave in wavelengths:
		print("Updating: {}".format(wave))
		
		###Download the newest image
		
		# Define the Image
		webPath = webString.format(wave)
		referencePath = normpath(pathToFile + wave + "_Now.jpg")
		
		# Define the Folder
		folderPath = normpath(pathToFile +wave)+"\\"
		pathlib.Path(folderPath).mkdir(parents=True, exist_ok=True)
		
		# Figure out which frames already exist
		try:
			items = listdir(folderPath)	
			
			framePaths = [folderPath + item for item in items]
			mTimes = [os.path.getmtime(pa) for pa in framePaths]
			lastFrameInd = np.argmax(mTimes)
			
			inds = [ int(item.split('.')[0].split('_')[1]) for item in items]
		
		
			maxInd = inds[lastFrameInd]
			newInd = np.mod(maxInd + 1, maxFrameCount)
			checkLast = True
		except: 
			newInd = 0
			checkLast = False
			
		
		# Define the filePath 
		fileEnd = "_{}.jpg".format(int(newInd))
		filePath = normpath(folderPath + wave + fileEnd)
			
		
		# Download the Image
		downloadImage(webPath, filePath)

		
		if checkLast:
			# Delete it if it is not new
			imgHourL, imgMinL, _ = readTime(framePaths[lastFrameInd], wave)
			imgHour, imgMin, _ = readTime(filePath, wave)
			
			if imgHourL == imgHour and imgMinL == imgMin:			
				os.remove(filePath)
			else:
				# Modify the Image
				imageMod(filePath, wave)

				# Copy it up a level for convenience
				copyfile(filePath, referencePath)
		else:
			# Modify the Image
			imageMod(filePath, wave)

			# Copy it up a level for convenience
			copyfile(filePath, referencePath)		
		########################################################

		###Display a loop for a duration
		framePaths = [folderPath + item for item in listdir(folderPath)]
		
		#Loop over all the frames
		
		startTime = time.time()
		
		print("Looping over {} ({} frames)...".format(wave, len(framePaths)), end = '')
		sys.stdout.flush()
		
		while np.abs(startTime - time.time()) < loopTime:
			for fPath in framePaths:
				
				# Update the Background
				updateBackground(fPath)

				# Wait for an instant
				time.sleep(frameTime)	
			time.sleep(frameTime)
		print("Done\n")
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
