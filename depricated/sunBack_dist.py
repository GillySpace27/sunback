from ctypes import windll
import time
import pathlib

from urllib.request import urlretrieve as urlret
from random import shuffle
import os.path as path

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

import pytesseract as tes
from numpy import mod
from numpy import floor
 
minutes = 60
hours =  minutes * 60

# pathToFile = r"C:\Users\chgi7364\Dropbox\Drive\Pictures\SunToday\\"

pathToFile = path.dirname(path.abspath(__file__)) + "\Images\\"
pathlib.Path(pathToFile).mkdir(parents=True, exist_ok=True)


webString = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_{}.jpg"
wavelengths = ['0171','0193','0211','0304','0131','0335','0094','HMIBC','HMIIF']
wavelengths.sort()
fileEnd = "_Now.jpg"


picChangeTime = 0.5 * minutes

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
		windll.user32.SystemParametersInfoW(20, 0, fullPath, 0)
		print("Background Updated\n")
	except:print("Failed to update background"); pass
	
def saveReference(pathToFile, wave):
	"""Grab a single reference image and save"""
	try:
		print("Saving Reference Image...", end ='')
		now = time.strftime("%y-%j-%H-%M")
		webpath = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_{}.jpg".format(wave)
		fullpath = path.normpath(pathToFile + "reference\\171_reference_{}.jpg".format(now))
		urlret(webpath, fullpath)
		print("Success")
	except: print("Failed")
	
def imageMod(fullPath, wave):
	"""Modify the image"""
	try: 
		print('Modifying Image...', end='')
		#Open the image for modification
		img = Image.open(fullPath)
		img_raw = img
		font = ImageFont.truetype(path.normpath("C:\Windows\Fonts\Arial.ttf"), 42)
		
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
		if time.localtime().tm_isdst: offset = time.altzone / hours
		else: offset = time.timezone / hours
		
		try:
			cropped = img_raw.crop((0,1950, 1024, 2048))
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
		#Draw on the image and save
		draw = ImageDraw.Draw(img)
		
		towrite = wave[1:] if wave[0] =='0' else wave
		
		draw.text((1510, 300),towrite,(200,200,200),font=font) 
		draw.rectangle([(450,150),(560,200)], fill=(0,0,0))
		draw.text((450, 150),time.strftime("%I:%M"),(200,200,200),font=font)
		draw.text((450, 300),"{:0>2}:{:0>2}{}".format(imgHour, imgMin, pre),(200,200,200),font=font)
		img.save(fullPath)
		print("Success")
	except: print("Failed")
	return
	
	
##The Main Loop

print("\nLive SDO Background Updater \nWritten by Gilly\n") 

while True:

	for wave in wavelengths:
		try:
			# Define the Image
			print("Image: {}".format(wave))
			webPath = webString.format(wave)
			fullPath = path.normpath(pathToFile + wave + fileEnd)
			
			#Download the Image
			downloadImage(webPath, fullPath)
			
			#Modify the Image
			imageMod(fullPath, wave)
			
			#Update the Background
			updateBackground(fullPath)
			
			#Wait for a bit
			time.sleep(picChangeTime)
		except (KeyboardInterrupt, SystemExit):print("Fine, I'll Stop.\n"); raise
		except: print("I failed")
	saveReference(pathToFile, '0211')
	
	
	
	
