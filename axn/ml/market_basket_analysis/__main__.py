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

    ######################################################################
    #
    # read run commands
    #
    args = parse_command_line()
    file_in_name = args.file_in
    file_out = args.file_out

    ######################################################################

    #
    #
    # pylint: disable=duplicate-code

    print("MBA --- START ")


    file_in_name2 = "/Users/tomlorenc/Downloads/Final_Dagger_Data_V9.csv"
    file_out_name2 = "/Users/tomlorenc/Downloads/Final_Dagger_Data_V9_MBA.csv"

    df2 = pd.read_csv(file_in_name2)
    print(df2.head())

    frequent_itemsets2 = apriori(df2, min_support=0.07, use_colnames=True)
    print("frequent_itemsets2 ... ")
    print(frequent_itemsets2)
    rules = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
    print(rules.head())

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
    rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")


    #for index, row in rules.iterrows():
    #    print("antecedents : " + str(row['antecedents']), "consequents : " + str(row['consequents']), str(row['support']), str(row['confidence']), str(row['lift']))

    rules.to_csv(file_out_name2)
    print("MBA --- END ")

if __name__ == '__main__':
    main()
