#!/usr/bin/env bash
        python -m discrete  \
          --file_in covid.csv \
          --drop_column  City\
  --drop_column  State\
  --drop_column  Country \
              --drop_column  6/4/20\
  --drop_column  6/5/20 \
            --dicretize uniform 4000 5\/31\/20 \
  --dicretize uniform 4000 6\/1\/20 \
  --dicretize uniform 4000 6\/2\/20 \
  --dicretize uniform 4000 6\/3\/20  \
          --file_out_discrete /Users/tomlorenc/Sites/VL_standard/ml/05-31-2020_06-03-2020/time_series_covid19_deaths_US_V4000_v1_i_D_05-31-2020_06-03-2020.csv \
          --file_out /Users/tomlorenc/Sites/VL_standard/ml/05-31-2020_06-03-2020/time_series_covid19_deaths_US_V4000_v3_D_05-31-2020_06-03-2020.csv