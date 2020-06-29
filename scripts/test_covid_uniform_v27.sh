#!/usr/bin/env bash


python -m predict  \
  --file_in csvs/covid_11_pivot_death_uniform_D_250_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target 4_26_20_deaths_DISCRETE \
  --training_test_split_percent 70 \
  --predictor RF \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/covid_11_pivot_death_uniform_D_250_OHE_PREDICT_RF.csv