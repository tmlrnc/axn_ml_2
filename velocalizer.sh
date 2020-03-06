#!/usr/bin/env bash

python -m velocalizer  \
  --file_in csvs/Married_Dog_Child_ID_Age_Home.csv \
  --file_out csvs/Married_Dog_Child_ID_Age_Home_SAMPLE.csv \
  --noise_threshold 30


