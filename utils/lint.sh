#!/bin/bash
set -e
CMD=$0
read -r -d '\0' DOCUMENTATION << EOF
Used for linting the project

Usage:
  $CMD lint - Runs the linting tool over the project

Environment Variables:
\0
EOF

function lint {
    pylint correlation
    pylint covid
    pylint discrete
    pylint feature_sample
    pylint generate_discrete
    pylint generate_master
    pylint generate_ohe
    pylint generate_predict
    pylint ohe
    pylint pipeline
    pylint predict
}

ACTION=$1

case "$ACTION" in
    lint) lint;;
  *)
  echo "$DOCUMENTATION"
  exit 1
esac
