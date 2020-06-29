#!/usr/bin/env bash


python -m discrete  \
  --file_in covid_joe8_pivot_death_org3.csv \
    --dicretize analyst_supervised   5  4_9_20_deaths 10 50 200 500 1500 3000 6000 10000 \
        --dicretize analyst_supervised   5  4_8_20_deaths 10 50 200 500 1500 3000 6000 10000 \
    --dicretize analyst_supervised   5  4_10_20_deaths 10 50 200 500 1500 3000 6000 10000 \
  --file_out_discrete csvs/covid_joe8_pivot_death_org3_v.csv \
  --file_out csvs/covid_joe8_pivot_death_org3_D.csv

