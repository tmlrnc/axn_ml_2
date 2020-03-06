import argparse
import csv
import math

import pandas

import numpy as np
def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in',
                        required=True,
                        help="The data file to velocalyze.")

    args = parser.parse_args()
    return args
def histogram_intersection(a, b):

    v = np.minimum(a, b).sum().round(decimals=1)

    return v


def main():
    """

    """
    args = parse_command_line()
    file_in_name = args.file_in
    noise_threshold = args.noise_threshold

    df = pandas.read_csv('csvs/Correlation.csv')

    my_corr = df.corr(method = 'kendall')
    print(my_corr)









    import pandas as pd
    import numpy as np
    filename = file_in_name
    nlinesfile = 100
    nlinesrandomsample = 50


    lines2skip = np.random.choice(np.arange(1, nlinesfile + 1), (nlinesfile - nlinesrandomsample), replace=False)
    print(lines2skip)
    df = pd.read_csv(filename, skiprows=lines2skip)
    print(df)




if __name__ == '__main__':
    main()
