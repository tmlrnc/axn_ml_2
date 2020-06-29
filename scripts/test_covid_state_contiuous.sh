#!/usr/bin/env bash



python -m ohe  \
  --file_in covid_joe9_pivot_death_org.csv \
  --ignore 4_9_20_deaths \
   --ignore 4_8_20_deaths \
  --ignore   4_26_20_deaths  \
  --ignore   4_25_20_deaths\
  --ignore   4_24_20_deaths  \
  --ignore   4_23_20_deaths  \
  --ignore   4_22_20_deaths  \
  --ignore   4_21_20_deaths\
  --ignore  4_20_20_deaths\
  --ignore   4_19_20_deaths  \
  --ignore  4_18_20_deaths  \
  --ignore   4_17_20_deaths  \
  --ignore   4_16_20_deaths  \
  --ignore   4_15_20_deaths  \
    --ignore   4_14_20_deaths  \
  --ignore   4_13_20_deaths  \
  --ignore   4_12_20_deaths  \
  --ignore   4_11_20_deaths  \
  --ignore   4_10_20_deaths  \
  --file_out csvs/covid_joe9_pivot_death_org_OHE.csv



python -m predict  \
  --file_in csvs/covid_joe9_pivot_death_org_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target 4_26_20_deaths \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/covid_joe9_pivot_death_org_OHE_PREDICT.csv

