import ctypes
import time

from urllib.request import urlretrieve as urlret
from random import shuffle
from os.path import normpath

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


import pytesseract as tes
from numpy import mod
from numpy import floor
 


pathToFile = r"C:\Users\chgi7364\Dropbox\Drive\Pictures\SunToday\\"
wavelengths = ['0171','0193','0211','0304','0131','0335','0094','1600','HMIBC','HMIIF']

fileEnd = "_Today.jpg"

lastTime = 0
minutes = 60
hours =  minutes * 60
picChangeTime = 16 * minutes / len(wavelengths)
refreshPicsTime = 10 * minutes
startTime = time.time()
while True:

	try:
		#Download All Pictures
		if time.time() - lastTime > refreshPicsTime:
			print("Downloading Images...")
			lastTime = time.time()
			for wave in wavelengths:
				# if wave[0]!='H':continue
				webPath = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_{}.jpg".format(wave)
				fullPath = normpath(pathToFile + wave + fileEnd)
				
				try:urlret(webPath, fullPath)
				except: urlret(webPath, fullPath)
				
				try: #Modify the image
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

					#Draw on the image and save
					draw = ImageDraw.Draw(img)
					
					towrite = wave[1:] if wave[0] =='0' else wave
					
					draw.text((1510, 300),towrite,(200,200,200),font=font) 
					
					draw.text((450, 300),"{:0>2}:{:0>2}{}".format(imgHour, imgMin, pre),(200,200,200),font=font)
					img.save(fullPath)
				except: print("I couldn't modify the Image"); pass
				
			#Grab a single reference image and save
			elapsed = time.strftime("%y-%j-%H-%M")
			webpath = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_2048_0171.jpg"
			fullpath = normpath(pathToFile + "reference\\171_reference_{}.jpg".format(elapsed))
			urlret(webpath, fullpath)
			print("Images Downloaded Successfully")

		#Set the Background	
		shuffle(wavelengths)
		for wave in wavelengths:
			fullPath = normpath(pathToFile + wave + fileEnd)
			
			img = Image.open(fullPath)
			draw = ImageDraw.Draw(img)
			draw.text((450, 150),time.strftime("%I:%M"),(200,200,200),font=font)
			img.save(fullPath)
			
			ctypes.windll.user32.SystemParametersInfoW(20, 0, fullPath, 0)
			print("Background Updated")
			time.sleep(picChangeTime)
	except: print("I failed"); continue

	
	
	
	
	
	
	
	
	
	
	
