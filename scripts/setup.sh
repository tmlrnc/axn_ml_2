#!/usr/bin/env bash

# NUKES THE INSTALLATION DIRECTORY SO YOU CAN START OVER. IF THINGS ARE NOT WORKING, TRY THIS.

rm -r ~/.ml
virtualenv -p python3 ~/.ml/
rm activate
ln -s ~/.ml/bin/activate .
source activate
pip install -r requirements.txt
echo 'Run `source activate` to enter environment.'