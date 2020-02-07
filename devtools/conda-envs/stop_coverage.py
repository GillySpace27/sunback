

import platform
import os

this_system = platform.system()

if this_system == "Windows":
    #Windows

    os.system("./cc-test-reporter after-build --exit-code $TRAVIS_TEST_RESULT")

elif this_system == "Linux":
    #Linux
    os.system("./cc-test-reporter after-build --exit-code $TRAVIS_TEST_RESULT")
    
elif this_system == "Darwin":
    #Mac
    pass
else:
    raise OSError("Operating System Not Supported")
