#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/14_TEST.csv \
  --dicretize uniform   3 POLLUTION_INDEX  \
  --dicretize uniform   3  POLLUTION_INDEX2  \
  --file_out_discrete csvs/14_TEST_D.csv \
  --file_out csvs/14_TEST_D_2.csv


python -m predict  \
  --file_in csvs/14_TEST_D_2.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target POLLUTION_INDEX2_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/14_TEST_D_2_PREDICT.csv


