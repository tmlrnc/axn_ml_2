#!/usr/bin/env bash

echo "Today is $(date)"
file_name=$(date +'%m-%d-%Y')
echo $file_name + ".csv"
file_name+=".csv"


python -m covid  \
  --url_in https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv \
  --file_out $file_name

