#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/17_POLL_MORT.csv \
  --dicretize uniform   3 POLLUTION_INDEX  \
  --dicretize uniform   3  MORT_RATIO  \
  --file_out_discrete csvs/17_TEST_D2.csv \
  --file_out csvs/17_TEST_D.csv


python -m predict  \
  --file_in csvs/17_TEST_D.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target MORT_RATIO_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/17_TEST_D_PREDICT.csv


