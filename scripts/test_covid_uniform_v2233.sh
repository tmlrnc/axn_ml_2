#!/usr/bin/env bash


python -m discrete  \
  --file_in time_series_covid19_deaths_US_V4.csv \
    --dicretize uniform   2000  4_1_20 \
        --dicretize uniform   2000  4_2_20 \
    --dicretize uniform   2000  4_3_20 \
    --dicretize uniform   2000  4_4_20 \
    --dicretize uniform   2000  4_5_20 \
    --dicretize uniform   2000  4_6_20 \
    --dicretize uniform   2000  4_7_20 \
    --dicretize uniform   2000  4_9_20 \
        --dicretize uniform   2000 4_8_20 \
    --dicretize uniform   2000 4_10_20  \
        --dicretize uniform   2000  4_11_20  \
    --dicretize uniform   2000  4_12_20  \
    --dicretize uniform   2000 4_13_20  \
    --dicretize uniform   2000  4_14_20  \
    --dicretize uniform   2000  4_15_20  \
    --dicretize uniform   2000  4_16_20  \
    --dicretize uniform   2000  4_17_20  \
    --dicretize uniform   2000 4_18_20  \
    --dicretize uniform   2000  4_19_20  \
        --dicretize uniform   2000  4_20_20  \
        --dicretize uniform   2000 4_21_20  \
        --dicretize uniform   2000  4_22_20  \
        --dicretize uniform   2000  4_23_20  \
        --dicretize uniform   2000  4_24_20  \
        --dicretize uniform   2000  4_25_20  \
        --dicretize uniform   2000    4_26_20  \
                --dicretize uniform   2000    4_27_20  \
        --dicretize uniform   2000    4_28_20  \
        --dicretize uniform   2000    4_29_20  \
        --dicretize uniform   2000    4_30_20  \
           --dicretize uniform   2000    5_1_20  \
           --dicretize uniform   2000    5_2_20  \
           --dicretize uniform   2000    5_3_20  \
           --dicretize uniform   2000    5_4_20  \
           --dicretize uniform   2000    5_5_20  \
           --dicretize uniform   2000    5_6_20  \
           --dicretize uniform   2000    5_7_20  \
           --dicretize uniform   2000    5_8_20  \
  --file_out_discrete csvs/time_series_covid19_deaths_US_V2000_i.csv \
  --file_out csvs/time_series_covid19_deaths_US_V2000_D.csv

