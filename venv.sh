#!/bin/bash

cwd=$(pwd)

# Below command creates a new virtual environment in the current working directory.
python -m venv "$cwd"
echo "Virtual environment has been created in directory - $cwd"

# Below command activates the virtual environment in the current directory.
source ./bin/activate

# Check if the output of this command points to python interpreter inside the venv.
which python

echo "Installing all the requirements now..."
pip install -r requirements.txt