"""
main for discrete
"""
# pylint: disable=unused-variable
# pylint: disable=line-too-long
# pylint: disable=duplicate-code

import argparse

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def parse_command_line():
    """
    reads the command line args
    """
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_in',
        help='raw csv file input to be predicted. Must be a csv file where first row has column header '
             'names. Must include time series date columns - like MM/DD/YY (7/3/20) ')
    parser.add_argument(
        '--file_out',
        help='csv file output encoded using one-hot one-of-K encoding scheme')
    args = parser.parse_args()
    return args



def main():
    """
Market Basket Analysis

also called Association analysis is light on the math concepts and easy to explain to non-technical people.
In addition, it is an unsupervised learning tool that looks for hidden patterns so there is
limited need for data prep and feature engineering.
It is a good start for certain cases of data exploration and can point the way for a deeper dive into the data using other approaches.

Association rules are normally written like this: {Diapers} -> {Beer} which means that there is a strong relationship between customers
that purchased diapers and also purchased beer in the same transaction.

      """
    # pylint: disable=duplicate-code

    args = parse_command_line()
    file_in_name2 = args.file_in
    file_out_name2 = args.file_out
    df2 = pd.read_csv(file_in_name2)

    df2['axn_CnsmBhvr_ArtsCul'] = df2['axn_CnsmBhvr_ArtsCul'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_HealthWell'] = df2['axn_CnsmBhvr_HealthWell'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_DIY'] = df2['axn_CnsmBhvr_DIY'].apply(lambda x: 0 if x < 6 else 1)


    df2.to_csv(file_out_name2)

    print("MBA --- END ")

if __name__ == '__main__':
    main()
