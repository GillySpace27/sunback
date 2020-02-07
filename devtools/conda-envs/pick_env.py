

import platform
import os

this_system = platform.system()

if this_system == "Windows":
    #Windows Imports
    os.system("python devtools/scripts/create_conda_env.py -n=test -p=$PYTHON_VER devtools/conda-envs/test_env.yaml")

elif this_system == "Linux":
    #Linux Imports
    os.system("python devtools/scripts/create_conda_env.py -n=test -p=$PYTHON_VER devtools/conda-envs/test_env.yaml")

elif this_system == "Darwin":
    #Mac Imports
    os.system("python devtools/scripts/create_conda_env.py -n=test -p=$PYTHON_VER devtools/conda-envs/test_env_mac.yaml")

else:
    raise OSError("Operating System Not Supported")
