#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/2020_04_08_agg_covid_data_pivot_6.csv \
  --dicretize uniform   60  deaths  \
  --file_out_discrete csvs/2020_04_08_agg_covid_data_pivot_6_D.csv \
  --file_in_plus_discrete csvs/2020_04_08_agg_covid_data_pivot_6_PLUS_D.csv


python -m ohe  \
  --file_in csvs/2020_04_08_agg_covid_data_pivot_6_PLUS_D.csv \
  --file_out_ohe csvs/2020_04_08_agg_covid_data_pivot_6_PLUS_D_OHE.csv \
  --file_out_predict csvs/2020_04_08_agg_covid_data_pivot_6_PLUS_D_OHE_PREDICT.csv \
  --file_in_config config/ohe_config_RUN1.yaml \
  --ignore deaths_DISCRETE \
  --ignore deaths \
  --target_ignore status \
  --target status \
  --training_test_split_percent 70 \
  --write_predictions YES \
  --predictor SVM \
  --predictor LR \
  --score f1_score \
  --score classification_accuracy \
  --score recall



