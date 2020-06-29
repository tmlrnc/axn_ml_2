#!/usr/bin/env bash


python -m discrete  \
  --file_in covid_10_pivot_death.csv \
    --dicretize dbscan   9  4_9_20_deaths \
        --dicretize dbscan   9  4_8_20_deaths \
    --dicretize dbscan   9  4_10_20_deaths  \
        --dicretize dbscan   9  4_11_20_deaths  \
    --dicretize dbscan   9  4_12_20_deaths  \
    --dicretize dbscan   9  4_13_20_deaths  \
    --dicretize dbscan   9  4_14_20_deaths  \
    --dicretize dbscan   9  4_15_20_deaths  \
    --dicretize dbscan  9  4_16_20_deaths  \
    --dicretize dbscan   9  4_17_20_deaths  \
    --dicretize dbscan   9  4_18_20_deaths  \
    --dicretize dbscan   9  4_19_20_deaths  \
        --dicretize dbscan   9 4_20_20_deaths  \
        --dicretize dbscan   9  4_21_20_deaths  \
        --dicretize dbscan   9  4_22_20_deaths  \
        --dicretize dbscan   9  4_23_20_deaths  \
        --dicretize dbscan   9  4_24_20_deaths  \
        --dicretize dbscan   9  4_25_20_deaths  \
        --dicretize dbscan   9  4_26_20_deaths  \
  --file_out_discrete csvs/covid_10_pivot_death_dbscan_31i.csv \
  --file_out csvs/covid_10_pivot_death_dbscan_D31.csv

