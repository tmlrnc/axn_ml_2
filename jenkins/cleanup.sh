#!/bin/bash
set -e
set -x
ENV_PATH=".python_env"
source .python_env/bin/activate
if [[ "$BRANCH_NAME" == *"develop"* ]]; then
    echo "Merged into develop, pruning buckets..."
    bash utils/doc.sh prune
fi
