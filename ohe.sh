#!/usr/bin/env bash

python -m ohe  \
  --file_in csvs/Married_Dog_Child_ID_Age.csv \
  --file_out_ohe csvs/Married_Dog_ID_Age_OHE.csv  \
  --file_out_predict csvs/Married_Dog_PREDICT_RUN1.csv \
  --file_in_config config/ohe_config_RUN1.yaml \
  --ignore ID \
  --ignore Age \
  --target Kids \
  --training_test_split_percent 70 \
  --predictor SVM \
  --predictor LR \
  --predictor RF \
  --predictor MLP \
  --predictor GPC \
  --predictor QDA \
  --predictor KNN \
  --predictor GNB \
  --predictor DTC \
  --predictor LDA \
  --predictor ETC \
  --predictor NU_SVM \
  --predictor ADA
