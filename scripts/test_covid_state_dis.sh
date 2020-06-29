#!/usr/bin/env bash


python -m discrete  \
  --file_in Covid_Death_State_Date_V2.csv \
  --dicretize uniform   400   4_26  \
    --dicretize uniform   400   4_25  \
  --dicretize uniform   400   4_24  \
  --dicretize uniform   400   4_23  \
  --dicretize uniform   400   4_22  \
  --dicretize uniform   400   4_21  \
  --dicretize uniform   400   4_20  \
  --dicretize uniform   400   4_19  \
    --dicretize uniform   400   4_18  \
  --dicretize uniform   400   4_17  \
  --dicretize uniform   400   4_16  \
  --dicretize uniform   400   4_15  \
  --dicretize uniform   400   4_14  \
  --dicretize uniform   400   4_13  \
  --file_out_discrete csvs/Covid_Death_State_Date_V4_v.csv \
  --file_out csvs/Covid_Death_State_Date_V4_D.csv

