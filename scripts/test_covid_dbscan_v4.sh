#!/usr/bin/env bash


python -m predict  \
  --file_in csvs/covid_10_pivot_death_D_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target 4_26_20_deaths_DISCRETE \
  --training_test_split_percent 70 \
  --predictor BAYESIANRIDGE \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/covid_10_pivot_death_D_OHE_PREDICT_MLP.csv