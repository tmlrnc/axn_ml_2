#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/11_2020_04_10_exp_countries_modeled_mort_V2.csv \
  --dicretize uniform   60  POLLUTION_INDEX  \
    --dicretize_many uniform   60  POLLUTION_INDEX  \
        --dicretize_many uniform   60  MORT_RATIO  \
  --file_out_discrete csvs/11_2020_04_10_exp_countries_modeled_mort_D.csv \
  --file_out csvs/11_PLUS_D.csv


python -m ohe  \
  --file_in csvs/11_PLUS_D.csv \
  --ignore POLLUTION_INDEX_DISCRETE \
   --ignore MORT_RATIO_DISCRETE \
  --file_out csvs/11_PLUS_D_OHE.csv


python -m predict  \
  --file_in csvs/11_PLUS_D_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config_RUN1.yaml \
  --target MORT_RATIO \
  --training_test_split_percent 70 \
  --predictor SVM \
  --predictor LR \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/11_PLUS_D_OHE_PREDICT.csv


