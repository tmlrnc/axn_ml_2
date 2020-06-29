#!/usr/bin/env bash


python -m discrete  \
  --file_in covid_data/2020-06-03-test2.csv \
     --drop_column UID \
          --drop_column iso2 \
                    --drop_column iso3 \
          --drop_column code3 \
          --drop_column FIPS \
          --drop_column Admin2 \
          --drop_column Province_State \
          --drop_column Country_Region \
          --drop_column Lat \
          --drop_column Long_\
          --drop_column Population\
      --dicretize uniform   4000  3\/24\/20 \
       --dicretize uniform   4000  3\/25\/20 \
      --dicretize uniform   4000  3\/26\/20 \
  --file_out_discrete csvs/2020-06-01-test_i.csv \
  --file_out csvs/2020-06-01-test_D_4000_v2.csv

