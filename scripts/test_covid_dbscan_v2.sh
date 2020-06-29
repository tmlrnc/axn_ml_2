#!/usr/bin/env bash


python -m discrete  \
  --file_in covid_10_pivot_death.csv \
    --dicretize dbscan   3  4_9_20_deaths \
        --dicretize dbscan   3  4_8_20_deaths \
    --dicretize dbscan   3  4_10_20_deaths  \
        --dicretize dbscan   3  4_11_20_deaths  \
    --dicretize dbscan   3  4_12_20_deaths  \
    --dicretize dbscan   3  4_13_20_deaths  \
    --dicretize dbscan   3  4_14_20_deaths  \
    --dicretize dbscan   3  4_15_20_deaths  \
    --dicretize dbscan   3  4_16_20_deaths  \
    --dicretize dbscan   3  4_17_20_deaths  \
    --dicretize dbscan   3  4_18_20_deaths  \
    --dicretize dbscan   3  4_19_20_deaths  \
        --dicretize dbscan   3  4_20_20_deaths  \
        --dicretize dbscan   3  4_21_20_deaths  \
        --dicretize dbscan   3  4_22_20_deaths  \
        --dicretize dbscan   3  4_23_20_deaths  \
        --dicretize dbscan   3  4_24_20_deaths  \
        --dicretize dbscan   3  4_25_20_deaths  \
        --dicretize dbscan   3  4_26_20_deaths  \
  --file_out_discrete csvs/covid_10_pivot_death_dbscan_i.csv \
  --file_out csvs/covid_10_pivot_death_dbscan_D.csv

