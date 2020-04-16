#!/usr/bin/env bash

python -m discrete  \
  --file_in csvs/C101.csv \
  --dicretize uniform   10  deaths  \
  --file_out_discrete csvs/C_D_102.csv \


