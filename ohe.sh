#!/usr/bin/env bash

python -m ohe  \
  --file_in csvs/Married_Dog_Child_ID_Age_Home.csv \
  --file_out_ohe csvs/Married_Dog_Child_ID_Age_Home_OHE.csv  \
  --file_out_predict csvs/Married_Dog_Child_ID_Age_Home_PRED.csv \
  --file_in_config config/ohe_config_RUN1.yaml \
  --ignore ID \
  --ignore Age \
  --target Kids \
  --training_test_split_percent 70 \
  --ohe_only NO \
  --predictor SVM \
  --predictor ADA \
  --predictor ENTROPY_DECISION_TREE \
  --predictor LeastSquaresLDA \
  --predictor MLP
