#!/usr/bin/env bash

python -m ohe  \
  --file_in csvs/COVID_IN_V99.csv \
  --file_out_discrete csvs/COVID-19-V6_DIS_V99.csv \
  --file_out_ohe csvs/COVID-19-V6_OHE_V99.csv \
  --file_out_ohe_dis csvs/COVID-19-V6_OHE_DIS100.csv \
  --file_out_predict csvs/COVID-19-V6_OHE_DIS_predict100.csv \
  --file_in_config config/ohe_config_RUN1.yaml \
  --ignore PPM \
  --ignore DEATH \
  --dicretize uniform   5  PPM  \
  --target STATUS \
  --training_test_split_percent 70 \
  --write_predictions YES \
  --predictor SVM \
  --predictor LR \
  --score f1_score \
  --score classification_accuracy \
  --score recall

