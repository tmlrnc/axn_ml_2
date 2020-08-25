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


In order to make it easier to understand, think of Market Basket Analysis in terms of shopping at a supermarket. Market Basket Analysis takes data at transaction level, which lists all items bought by a customer in a single purchase. The technique determines relationships of what products were purchased with which other product(s). These relationships are then used to build profiles containing If-Then rules of the items purchased.

The rules could be written as:

If {Antecedent} Then {Consequent}

The If part of the rule (the {A} above) is known as the antecedent and the THEN part of the rule is known as the consequent (the {Consequent} above). The antecedent is the condition and the consequent is the result. The association rule has three measures that express the degree of confidence in the rule, Support, Confidence, and Lift.

For example, you are in a supermarket to buy milk. Based on the analysis, are you more likely to buy apples or cheese in the same transaction than somebody who did not buy milk?


Support ( A for C)  = Number of Transaction with BOTH Antecedent AND Consequent ONLY / Total Number of ALL Transactions

SUPPORT is how frequent an Antecedent is in all the transactions
SUPPORT = (Num Transactions with Antecedent AND Consequent )/Total Num Transaction


CONFIDENCE = (Num Transactions with Antecedent AND Consequent )/ Num Transactions with Antecedent
CONFIDENCE is likeliness of occurrence of Consequent Given the Antecedent


      """
    # pylint: disable=duplicate-code

    args = parse_command_line()
    file_in_name2 = args.file_in
    file_out_name2 = args.file_out
    df2 = pd.read_csv(file_in_name2)
    import numpy as np


    consequent = 'JewelryCoverqage_Y'
    total_num_consequent = 0
    #support
    allrows = []
    for col in df2.columns:
        print(col)


        col_count = np.count_nonzero(df2[col])
        my_row = [col, col_count]
        if col == consequent:
            total_num_consequent = col_count
        allrows.append(my_row)


    allrows.sort(key=lambda x: x[1],reverse=True)

    import csv

    print("mycount")
    final_rows = []
    first_row = [""]

    mycount = df2.count()
    with open(file_out_name2, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(allrows)




if __name__ == '__main__':
    main()
