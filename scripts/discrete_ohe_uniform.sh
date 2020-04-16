#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/C102.csv \
  --dicretize uniform   10  deaths  \
  --file_out_discrete csvs/C102_D.csv \
  --file_in_plus_discrete csvs/C102_PLUS_D.csv


python -m ohe  \
  --file_in csvs/C102_PLUS_D.csv \
  --file_out_ohe csvs/C102_PLUS_D_OHE.csv \
  --file_out_ohe_dis csvs/C102_PLUS_D_OHE_2.csv \
  --file_out_predict csvs/C102_PLUS_D_OHE_PREDICT.csv \
  --file_in_config config/ohe_config_RUN1.yaml \
  --ignore recovered \
  --ignore deathsDISCRETE \
  --ignore deaths \
  --ignore confirmed \
  --target STATUS \
  --training_test_split_percent 70 \
  --write_predictions YES \
  --predictor SVM \
  --predictor LR \
  --score f1_score \
  --score classification_accuracy \
  --score recall



