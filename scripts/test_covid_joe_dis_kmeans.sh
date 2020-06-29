#!/usr/bin/env bash


python -m discrete  \
  --file_in covid_joe8_pivot_death_org3_KMEANS.csv \
    --dicretize kmeans   11  4_9_20_deaths \
        --dicretize kmeans   11  4_8_20_deaths \
    --dicretize kmeans   11  4_10_20_deaths  \
  --file_out_discrete csvs/covid_joe8_pivot_death_org3_KMEANS_v.csv \
  --file_out csvs/covid_joe8_pivot_death_org3_KMEANS_D.csv

