#!/usr/bin/env bash



python -m pipeline  \
  --url_in https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv   \
  --file_out csvs/time_series_covid19_deaths_US_DAILY_May_19.csv