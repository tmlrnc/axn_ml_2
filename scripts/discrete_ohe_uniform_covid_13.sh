#!/usr/bin/env bash



python -m predict  \
  --file_in csvs/13_PLUS_D_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target MORT_RATIO_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/13_PLUS_D_OHE_PREDICT.csv


