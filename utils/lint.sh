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
    pylint axn/ml/discrete
    pylint axn/ml/market_basket_analysis
    pylint axn/ml/ohe
    pylint axn/ml/predict
    pylint axn/ml/time_series


}

ACTION=$1

case "$ACTION" in
    lint) lint;;
  *)
  echo "$DOCUMENTATION"
  exit 1
esac
