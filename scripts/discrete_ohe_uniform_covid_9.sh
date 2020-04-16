#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/2020_04_08_agg_covid_data_pivot_9.csv \
  --dicretize uniform   60  deaths  \
  --file_out_discrete csvs/2020_04_08_agg_covid_data_pivot_9_D.csv \
  --file_out csvs/9_PLUS_D.csv


python -m ohe  \
  --file_in csvs/9_PLUS_D.csv \
  --ignore deaths_DISCRETE \
   --ignore status \
  --file_out csvs/9_PLUS_D_OHE.csv


python -m predict  \
  --file_in csvs/9_PLUS_D_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config_RUN1.yaml \
  --target status \
  --training_test_split_percent 70 \
  --predictor SVM \
  --predictor LR \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/9_PLUS_D_OHE_PREDICT.csv


