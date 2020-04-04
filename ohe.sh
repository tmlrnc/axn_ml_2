#!/usr/bin/env bash

python -m ohe  \
  --file_in csvs/COVID-19-V6.csv \
  --file_out_ohe csvs/Married_Dog_Child_ID_Age_Home_OHE.csv  \
  --file_out_predict csvs/Married_Dog_Child_ID_Age_Home_PRED.csv \
  --file_in_config config/ohe_config_RUN1.yaml \
  --ignore ID \
  --ignore Age \
  --dicretize_uniform PPM  \
  --target Death \
  --training_test_split_percent 70 \
  --ohe_only NO \
  --predictor SVM \
  --predictor ADA \
  --predictor LR \
  --predictor GNB \
  --predictor QDA \
  --predictor ENTROPY_DECISION_TREE \
  --predictor MLP \
  --score f1_score \
  --score classification_accuracy \
  --score recall

