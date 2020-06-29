#!/usr/bin/env bash


python -m discrete  \
  --file_in covid_joe8_pivot_death_org3_KMEANS.csv \
  --dicretize uniform   400   4_9_20_deaths  \
    --dicretize uniform   400   4_8_20_deaths  \
  --dicretize uniform   400   4_26_20_deaths  \
  --dicretize uniform   400   4_25_20_deaths  \
  --dicretize uniform   400   4_24_20_deaths  \
  --dicretize uniform   400   4_23_20_deaths  \
  --dicretize uniform   400   4_22_20_deaths  \
  --dicretize uniform   400   4_21_20_deaths  \
    --dicretize uniform   400   4_20_20_deaths  \
  --dicretize uniform   400   4_19_20_deaths  \
  --dicretize uniform   400   4_18_20_deaths  \
  --dicretize uniform   400   4_17_20_deaths  \
  --dicretize uniform   400   4_16_20_deaths  \
  --dicretize uniform   400   4_15_20_deaths  \
    --dicretize uniform   400   4_14_20_deaths  \
  --dicretize uniform   400   4_13_20_deaths  \
  --dicretize uniform   400   4_12_20_deaths  \
  --dicretize uniform   400   4_11_20_deaths  \
  --dicretize uniform   400   4_10_20_deaths  \
  --file_out_discrete csvs/covid_joe8_pivot_death_org1_v.csv \
  --file_out csvs/covid_joe8_pivot_death_org1_D.csv

