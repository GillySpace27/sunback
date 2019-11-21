import ctypes

def updateBackground(fullPath):
	"""Update the system background"""

	ctypes.windll.user32.SystemParametersInfoW(20, 0, fullPath, 0)
	# print("Background Updated\n")
# filename = "latest_1024_0304.mp4"

# updateBackground(os.path.normpath(pathToFile + filename))
	

	
import os
import pathlib
	
pathToFile = r"C:\Users\chgi7364\Dropbox\Drive\Pictures\SunToday\\"

wave = '0094sm'

folderPath = os.path.normpath(pathToFile +wave)+"\\"
pathlib.Path(folderPath).mkdir(parents=True, exist_ok=True)

import os
import ctypes
import time

#Get the filepaths of all pictures in a folder
items = os.listdir(folderPath)	
listOfFiles = [folderPath + item for item in items]

frameTime = 0.25 #second
runFor = 300 #seconds

#Cycle through the pictures 
startTime = time.time()
while time.time() - startTime < runFor:
    for file in listOfFiles:
        ctypes.windll.user32.SystemParametersInfoW(20, 0, file , 0)
        print("Thats a LooP")
        time.sleep(frameTime)