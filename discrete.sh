#!/usr/bin/env bash

python -m discrete  \
  --file_in csvs/COVID_IN_V99.csv \
  --dicretize kmeans   3  PPM   \
  --file_out_discrete csvs/COVID-19-V6_DIS_V99.csv


