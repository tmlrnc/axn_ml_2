#!/bin/bash
set -e
set -x
ENV_PATH=".python_env"
source .python_env/bin/activate
bash utils/lint.sh lint

