

import platform
import os

this_system = platform.system()

if this_system == "Windows":
    #Windows

    os.system("curl -L https://codeclimate.com/downloads/test-reporter/test-reporter-latest-linux-amd64 > ./cc-test-reporter")
    os.system("chmod 777 ./cc-test-reporter")
    os.system("./cc-test-reporter before-build")

elif this_system == "Linux":
    #Linux
    os.system("curl -L https://codeclimate.com/downloads/test-reporter/test-reporter-latest-linux-amd64 > ./cc-test-reporter")
    os.system("chmod 777 ./cc-test-reporter")
    os.system("./cc-test-reporter before-build")

elif this_system == "Darwin":
    #Mac
    pass
else:
    raise OSError("Operating System Not Supported")
