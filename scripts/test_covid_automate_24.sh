#!/usr/bin/env bash

if [[ -z $PARENT_DIR ]]; then
    echo "PARENT_DIR not defined"
    exit 0
fi

#echo "Today is $(date)"
file_name=$(date +'%m-%d-%Y')
#echo $file_name + ".csv"


file_name="time_series_covid19_deaths_US.csv"
parent_dir="$PARENT_DIR"

python -m generate_discrete  \
  --file_in $file_name \
          --drop_column Admin2 \
                    --drop_column code3 \
                    --parent_dir $parent_dir \
          --drop_column iso3 \
          --drop_column iso2 \
          --drop_column Country_Region \
          --drop_column FIPS \
                    --drop_column Lat \
          --drop_column Long_ \
          --drop_column Population \
   --file_out_discrete time_series_covid19_deaths_US_V4000_v1_i_D.csv \
  --file_out time_series_covid19_deaths_US_V4000_v3_D.csv \
  --start_date_all 07/01/2020 \
  --end_date_all 07/15/2020 \
  --num_bins 4000 \
  --window_size 3 \
  --discrete_file_script_out covid_discrete_test_15.sh

python -m generate_ohe  \
  --file_in time_series_covid19_deaths_US_V4000_v3_D.csv \
                     --parent_dir $parent_dir \
  --file_out  time_series_covid19_deaths_US_V4000_v3_OHE.csv \
  --start_date_all 07/01/2020 \
  --end_date_all 07/15/2020 \
    --window_size 3 \
    --ignore UID \
  --ohe_file_script_out covid_ohe_test_15.sh



python -m generate_predict  \
  --file_in time_series_covid19_deaths_US_V4000_v3_OHE.csv \
  --file_in_master $file_name \
    --target 7/14/20_DISCRETE \
   --add_model MLP \
                     --parent_dir $parent_dir \
  --start_date_all 07/01/2020 \
  --end_date_all 07/15/2020 \
    --window_size 3 \
            --ignore UID \
  --file_out_predict  time_series_covid19_deaths_US_V4000_v3_D_P.csv \
    --file_out_scores  time_series_covid19_deaths_US_V4000_v3_D_S.csv \
   --file_out_tableau tableau-file_out_master.csv  \
  --predict_file_script_out covid_predict_test_15.sh



python -m generate_master  \
  --file_in $file_name \
                     --parent_dir $parent_dir \
  --start_date_all 07/01/2020 \
  --end_date_all 07/15/2020 \
    --window_size 3 \
      --discrete_file_script_out covid_discrete_test_15.sh \
  --ohe_file_script_out covid_ohe_test_15.sh \
  --predict_file_script_out covid_predict_test_15.sh \
  --master_file_script_out covid_master_test_15.sh


echo "DONE"


