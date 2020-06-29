import argparse

import csv

import pandas
import pandas as pd
import numpy
import luigi




class MyTask(luigi.Task):
    x = luigi.IntParameter(default=45)
    y = luigi.IntParameter(default=45)

    def run(self):


        url_in = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
        fo= "csvs/time_series_covid19_deaths_US_DAILY_May_20.csv"
        data_frame_covid = pandas.read_csv(url_in).fillna(value=0)
        data_frame_covid.to_csv(fo)
        print("hi")
