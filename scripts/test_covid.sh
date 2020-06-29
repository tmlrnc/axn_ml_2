#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/2020_04_10_agg_covid_data_pivot_10.csv \
  --dicretize uniform   60  deaths_4_8  \
  --file_out_discrete csvs/2020_04_10_agg_covid_data_pivot_10_D.csv \
  --file_out csvs/10_PLUS_D.csv


python -m ohe  \
  --file_in csvs/10_PLUS_D.csv \
  --ignore deaths_4_8_DISCRETE \
   --ignore status \
  --file_out csvs/10_PLUS_D_OHE.csv


python -m predict  \
  --file_in csvs/10_PLUS_D_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config_RUN1.yaml \
  --target status \
  --training_test_split_percent 70 \
  --predictor SVM \
  --predictor LR \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/10_PLUS_D_OHE_PREDICT.csv

