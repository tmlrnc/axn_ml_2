#!/usr/bin/env bash

echo "Today is $(date)"
file_name=$(date +'%m-%d-%Y')
echo $file_name + ".csv"


file_name="06-15-2020-test17.csv"

python -m generate_discrete  \
  --file_in $file_name \
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
   --drop_column Population \
   --file_out_discrete time_series_covid19_deaths_US_V4000_v1_i_D.csv \
  --file_out time_series_covid19_deaths_US_V4000_v3_D.csv \
  --start_date_all 05/31/2020 \
  --end_date_all 06/14/2020 \
  --num_bins 4000 \
  --window_size 4 \
  --discrete_file_script_out covid_discrete_test_15.sh

python -m generate_ohe  \
  --file_in time_series_covid19_deaths_US_V4000_v3_D.csv \
  --file_out  time_series_covid19_deaths_US_V4000_v3_OHE.csv \
  --start_date_all 05/31/2020 \
  --end_date_all 06/14/2020 \
    --window_size 4 \
  --ohe_file_script_out covid_ohe_test_15.sh



python -m generate_predict  \
  --file_in time_series_covid19_deaths_US_V4000_v3_OHE.csv \
    --target 6/3/20_DISCRETE \
   --start_date_all 05/31/2020 \
   --add_model SVM \
   --add_model MLP \
   --add_model RFR \
  --end_date_all 06/14/2020 \
    --window_size 4 \
  --file_out_predict  time_series_covid19_deaths_US_V4000_v3_D_P.csv \
    --file_out_scores  time_series_covid19_deaths_US_V4000_v3_D_S.csv \
  --predict_file_script_out covid_predict_test_15.sh



python -m generate_master  \
  --file_in $file_name \
   --start_date_all 05/31/2020 \
  --end_date_all 06/14/2020 \
    --window_size 4 \
      --discrete_file_script_out covid_discrete_test_15.sh \
  --ohe_file_script_out covid_ohe_test_15.sh \
  --predict_file_script_out covid_predict_test_15.sh \
  --master_file_script_out covid_master_test_15.sh



