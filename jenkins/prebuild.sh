#!/bin/bash
set -e
set -x
ENV_PATH=".python_env"
rm -rf "$ENV_PATH"
virtualenv -p python3 "$ENV_PATH"
source "$ENV_PATH/bin/activate"
pip install -r requirements.txt

