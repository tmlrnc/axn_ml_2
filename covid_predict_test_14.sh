#!/usr/bin/env bash
        python -m predict  \
          --file_in /Users/tomlorenc/Sites/VL_standard/ml/06-10-2020_06-14-2020/time_series_covid19_deaths_US_V4000_v3_OHE_06-10-2020_06-14-2020.csv \
          --strategy none \
            --target   6\/14\/20_DISCRETE \
          --training_test_split_percent 70 \
               --predictor   SVM\
  --predictor   MLP\
  --predictor   RFR \
          --score f1_score \
          --score classification_accuracy \
          --score recall  \
          --file_in_config config/ohe_config.yaml \
          --file_out_scores /Users/tomlorenc/Sites/VL_standard/ml/06-10-2020_06-14-2020/time_series_covid19_deaths_US_V4000_v3_D_S_06-10-2020_06-14-2020.csv \
          --file_out_predict /Users/tomlorenc/Sites/VL_standard/ml/06-10-2020_06-14-2020/time_series_covid19_deaths_US_V4000_v3_D_P_06-10-2020_06-14-2020.csv