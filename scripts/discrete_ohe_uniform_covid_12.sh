#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/12_2020_04_10_exp_countries_modeled_mort.csv \
  --dicretize uniform   10  POLLUTION_INDEX  \
  --dicretize uniform   10  POLLUTION_INDEX2  \
  --file_out_discrete csvs/11_2020_04_10_exp_countries_modeled_mort_D.csv \
  --file_out csvs/12_PLUS_D.csv


python -m predict  \
  --file_in csvs/12_PLUS_D.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target POLLUTION_INDEX2_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/12_PLUS_D_OHE_PREDICT.csv


