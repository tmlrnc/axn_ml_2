import argparse

import csv

import pandas
import pandas as pd
import numpy
import luigi


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url_in')
    parser.add_argument('--file_out')
    args = parser.parse_args()
    return args





def main():
    args = parse_command_line()
    url_in = args.url_in
    file_out = args.file_out

    print("PIPELINE --- START ")

    data_frame_covid = pandas.read_csv(url_in).fillna(value=0)
    data_frame_covid.to_csv(file_out)



if __name__ == '__main__':
    main()


