#!/usr/bin/env bash




python -m predict  \
  --file_in csvs/2020-06-01-test_D_4000_OHE.csv \
  --strategy none \
  --file_in_config config/ohe_config.yaml \
  --target 4/7/20_DISCRETE \
  --training_test_split_percent 70 \
  --predictor SVM \
  --score f1_score \
  --score classification_accuracy \
  --score recall \
  --file_out_scores csvs/2020-06-01-test_D_4000_OHE_PREDICT_SCORE.csv \
  --file_out_predict csvs/2020-06-01-test_D_4000_OHE_PREDICT.csv