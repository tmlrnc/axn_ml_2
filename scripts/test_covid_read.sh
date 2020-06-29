#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/covid_date_p2.csv \
  --dicretize uniform   60  deaths_4_8  \
  --file_out_discrete csvs/covid_date_p2_out_d.csv \
  --file_out csvs/covid_date_p2_out.csv

