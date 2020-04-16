#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/19_POLL_MORT.csv \
  --dicretize uniform   3 POLLUTION_INDEX  \
  --dicretize uniform   3  MORT_RATIO  4 5 6 \
  --file_out_discrete csvs/18_POLL_MORT_D_I.csv \
  --file_out csvs/18_POLL_MORT_D.csv


python -m predict  \
  --file_in csvs/18_POLL_MORT_D_4_6.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target MORT_RATIO_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/18_POLL_MORT_D_PREDICT_4_6.csv


