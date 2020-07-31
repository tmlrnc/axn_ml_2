#!/bin/bash
set -e
set -x
ENV_PATH=".python_env"
source .python_env/bin/activate
bash utils/doc.sh generate
if [[ "$BRANCH_NAME" == *"PR"* ]]; then
   echo "This is a PR, skipping deployment of documentation."
   exit
fi
BUCKET_NAME="$BRANCH_NAME" bash utils/doc.sh deploy
