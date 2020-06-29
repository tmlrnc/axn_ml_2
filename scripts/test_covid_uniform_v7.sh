#!/usr/bin/env bash




python -m predict  \
  --file_in time_series_covid19_deaths_US_V4.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target  5_8_20 \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/time_series_covid19_deaths_US_V4_PREDICT.csv