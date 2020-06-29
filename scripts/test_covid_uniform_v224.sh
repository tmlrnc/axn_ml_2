#!/usr/bin/env bash


python -m ohe  \
  --file_in csvs/time_series_covid19_deaths_US_V44_D.csv \
     --ignore     4_1_20_DISCRETE \
    --ignore     4_2_20_DISCRETE \
    --ignore     4_3_20_DISCRETE \
    --ignore     4_4_20_DISCRETE \
    --ignore     4_5_20_DISCRETE \
    --ignore     4_6_20_DISCRETE \
    --ignore     4_7_20_DISCRETE \
    --ignore     4_9_20_DISCRETE \
        --ignore  4_8_20_DISCRETE \
    --ignore  4_10_20_DISCRETE   \
        --ignore 4_11_20_DISCRETE   \
    --ignore 4_12_20_DISCRETE   \
    --ignore  4_13_20_DISCRETE   \
    --ignore  4_14_20_DISCRETE   \
    --ignore  4_15_20_DISCRETE   \
    --ignore  4_16_20_DISCRETE   \
    --ignore  4_17_20_DISCRETE  \
    --ignore  4_18_20_DISCRETE  \
    --ignore  4_19_20_DISCRETE  \
        --ignore  4_20_20_DISCRETE  \
        --ignore  4_21_20_DISCRETE  \
        --ignore  4_22_20_DISCRETE  \
        --ignore  4_23_20_DISCRETE  \
        --ignore  4_24_20_DISCRETE  \
        --ignore  4_25_20_DISCRETE  \
        --ignore  4_26_20_DISCRETE  \
                --ignore  4_27_20_DISCRETE  \
        --ignore  4_28_20_DISCRETE  \
        --ignore  4_29_20_DISCRETE  \
        --ignore  4_30_20_DISCRETE  \
        --ignore  5_1_20_DISCRETE  \
        --ignore  5_2_20_DISCRETE  \
        --ignore  5_3_20_DISCRETE  \
        --ignore  5_4_20_DISCRETE  \
        --ignore  5_5_20_DISCRETE  \
        --ignore  5_6_20_DISCRETE  \
        --ignore  5_7_20_DISCRETE  \
        --ignore  5_8_20_DISCRETE  \
  --file_out csvs/time_series_covid19_deaths_US_V444_D_OHE.csv

python -m predict  \
  --file_in csvs/time_series_covid19_deaths_US_V444_D_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target 5_8_20_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/time_series_covid19_deaths_US_V444_D_OHE_PREDICT.csv