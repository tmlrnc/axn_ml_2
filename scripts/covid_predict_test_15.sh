#!/usr/bin/env bash
        python -m predict  \
          --file_in /Users/tomlorenc/Sites/VL_standard/ml/07-11-2020_07-14-2020/time_series_covid19_deaths_US_V4000_v3_OHE_07-11-2020_07-14-2020.csv \
                    --file_in_master time_series_covid19_deaths_US.csv \
          --strategy none \
            --target   7\/14\/20_DISCRETE \
         --ignore  UID \
          --training_test_split_percent 70 \
               --predictor   MLP \
          --score f1_score \
          --score classification_accuracy \
          --score recall  \
          --file_in_config config/ohe_config.yaml \
          --file_out_scores /Users/tomlorenc/Sites/VL_standard/ml/07-11-2020_07-14-2020/time_series_covid19_deaths_US_V4000_v3_D_S_07-11-2020_07-14-2020.csv \
                    --file_out_scores /Users/tomlorenc/Sites/VL_standard/ml/07-11-2020_07-14-2020/time_series_covid19_deaths_US_V4000_v3_D_S_07-11-2020_07-14-2020.csv \
        --file_out_tableau tableau-file_out_master.csv \
          --file_out_predict /Users/tomlorenc/Sites/VL_standard/ml/07-11-2020_07-14-2020/time_series_covid19_deaths_US_V4000_v3_D_P_07-11-2020_07-14-2020.csv