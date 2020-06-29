#!/usr/bin/env bash


python -m discrete  \
  --file_in csvs/2020_04_08_agg_covid_data_pivot_4.csv \
  --dicretize uniform   400  9_deaths  \
    --dicretize uniform   400  8_deaths  \
  --dicretize uniform   400  7_deaths  \
  --file_out_discrete csvs/covid_date_p4_out_d.csv \
  --file_out csvs/2020_04_08_agg_covid_data_pivot_4_out.csv


python -m ohe  \
  --file_in csvs/2020_04_08_agg_covid_data_pivot_4_out.csv \
  --ignore 9_deaths_DISCRETE \
    --ignore 8_deaths_DISCRETE \
  --ignore 7_deaths_DISCRETE \
  --file_out csvs/2020_04_08_agg_covid_data_pivot_4_out_OHE.csv


python -m predict  \
  --file_in csvs/2020_04_08_agg_covid_data_pivot_4_out_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target 9_deaths_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --predictor NUSVMSIG \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/2020_04_08_agg_covid_data_pivot_4_out_PREDICT.csv
