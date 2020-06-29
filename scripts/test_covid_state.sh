#!/usr/bin/env bash




python -m ohe  \
  --file_in csvs/Covid_Death_State_Date_V4_D.csv \
  --ignore 4_26_DISCRETE \
   --ignore 4_25_DISCRETE \
     --ignore 4_24_DISCRETE \
   --ignore 4_23_DISCRETE \
   --ignore 4_22_DISCRETE \
   --ignore 4_21_DISCRETE \
   --ignore 4_20_DISCRETE \
   --ignore 4_19_DISCRETE \
   --ignore 4_18_DISCRETE \
   --ignore 4_17_DISCRETE \
   --ignore 4_16_DISCRETE \
   --ignore 4_15_DISCRETE \
   --ignore 4_14_DISCRETE \
    --ignore 4_13_DISCRETE \
  --file_out csvs/Covid_Death_State_Date_V8_OHE.csv


python -m predict  \
  --file_in csvs/Covid_Death_State_Date_V8_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target 4_26_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out csvs/Covid_Death_State_Date_V8_OHE_PREDICT.csv

