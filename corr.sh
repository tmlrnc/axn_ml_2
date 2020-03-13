#!/usr/bin/env bash

python -m correlation  \
  --file_in csvs/Correlation_Married_Dog.csv \
  --file_out csvs/Correlation_Matrix.csv \
  --noise_threshold 3
