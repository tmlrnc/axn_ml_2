#!/usr/bin/env bash

echo "Today is $(date)"
file_name=$(date +'%m-%d-%Y')
echo $file_name + ".csv"


file_name="06-15-2020-test17.csv"





python -m generate_master  \
  --file_in $file_name \
   --start_date_all 05/31/2020 \
  --end_date_all 06/15/2020 \
    --window_size 4 \
      --discrete_file_script_out covid_discrete_test_15.sh \
  --ohe_file_script_out covid_ohe_test_15.sh \
  --predict_file_script_out covid_predict_test_15.sh \
  --master_file_script_out covid_master_test_15.sh



