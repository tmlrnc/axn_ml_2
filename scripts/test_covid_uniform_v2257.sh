#!/usr/bin/env bash


python -m predict  \
  --file_in csvs/time_series_covid19_deaths_US_V4000_D_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target 5_8_20_DISCRETE \
  --training_test_split_percent 70 \
  --predictor RF \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/time_series_covid19_deaths_US_V4000_D_OHE_PREDICT_RF.csv