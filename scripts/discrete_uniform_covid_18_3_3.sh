#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/18_POLL_MORT.csv \
  --dicretize uniform   3 POLLUTION_INDEX  \
  --dicretize uniform   3  MORT_RATIO  \
  --file_out_discrete csvs/18_POLL_MORT_D_I_3_3.csv \
  --file_out csvs/18_POLL_MORT_D_3_3.csv


