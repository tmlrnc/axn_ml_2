#!/usr/bin/env bash





python -m ohe  \
  --file_in csvs/2020_04_08_agg_covid_data_pivot_5_out.csv \
  --ignore 9_deaths_DISCRETE \
    --ignore 8_deaths_DISCRETE \
  --ignore 7_deaths_DISCRETE \
    --ignore 6_deaths_DISCRETE \
  --ignore 5_deaths_DISCRETE \
  --ignore 4_deaths_DISCRETE \
  --ignore 3_deaths_DISCRETE \
  --ignore 2_deaths_DISCRETE \
  --ignore 16_deaths_DISCRETE \
  --ignore 14_deaths_DISCRETE \
  --ignore 13_deaths_DISCRETE \
  --ignore 12_deaths_DISCRETE \
  --ignore 11_deaths_DISCRETE \
  --ignore 10_deaths_DISCRETE \
  --file_out csvs/2020_04_08_agg_covid_data_pivot_5_out_OHE.csv


python -m predict  \
  --file_in csvs/2020_04_08_agg_covid_data_pivot_5_out_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target 14_deaths_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/2020_04_08_agg_covid_data_pivot_5_out_PREDICT.csv
