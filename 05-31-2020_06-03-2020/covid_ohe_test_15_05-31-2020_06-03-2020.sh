#!/usr/bin/env bash


        python -m ohe  \
          --file_in /Users/tomlorenc/Sites/VL_standard/ml/05-31-2020_06-03-2020/time_series_covid19_deaths_US_V4000_v3_D_05-31-2020_06-03-2020.csv \
            --ignore   5\/31\/20_DISCRETE\
  --ignore   6\/1\/20_DISCRETE\
  --ignore   6\/2\/20_DISCRETE\
  --ignore   6\/3\/20_DISCRETE \
          --ignore   UID \
          --file_out /Users/tomlorenc/Sites/VL_standard/ml/05-31-2020_06-03-2020/time_series_covid19_deaths_US_V4000_v3_OHE_05-31-2020_06-03-2020.csv