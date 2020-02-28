#!/usr/bin/env bash

python -m velocalizer  \
  --file_in csvs/Married_Dog_Child_ID_Age_Home.csv \
  --noise_threshold 30
  #--file_out_ohe csvs/Married_Dog_ID_Age_OHE.csv  \
  #--file_out_predict csvs/Married_Dog_Home_PREDICT_All.csv

