"""
main for discrete

Market Basket Analysis REPORT for Association Rules



Support ( A for C)  = Number of Transaction with BOTH Antecedent AND Consequent ONLY / Total Number of ALL Transactions

SUPPORT is how frequent an Antecedent is in all the transactions
SUPPORT = (Num Transactions with Antecedent AND Consequent )/Total Num Transaction

CONFIDENCE is likeliness of occurrence of Consequent Given the Antecedent
CONFIDENCE = (Num Transactions with Antecedent AND Consequent )/ Num Transactions with Antecedent



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
        '--col_in',
        help='csv file output encoded using one-hot one-of-K encoding scheme')
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
    col_in = args.col_in
    file_in_name2 = args.file_in

    col_in_df = pd.read_csv(col_in)

    import numpy as np

    mycols = col_in_df.columns


    file_out_name2 = args.file_out
    consequent = 'JewelryCoverqage_Y'

    import csv

    with open(file_in_name2, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)


    total_num_transactions = 0
    new_list = []
    first_row = ["Consequent", "Antecedent", "Antecedent-Root", "Antecedent-Number", "Count", "Support", "Confidence"]
    #first_row = ["", "", "", "", "", "how frequent an Antecedent is", "likeliness of Consequent Given the Antecedent"]

    new_list.append(first_row)

    for row in data:
        print(str(row[0]))
        ant_full = str(row[0])
        if ant_full == consequent:
            total_num_transactions = row[1]
        for col in mycols:
            result = ant_full.find(col)
            if result != -1:
                print(col)
                mylen = len(col)
                print(mylen)
                ant_base = ant_full[0:mylen]
                ant_num = ant_full[mylen+1:]
                print("BASE " + str(ant_base))
                print("NUM " + str(ant_num))
                if ant_num == "0":
                    print("ERASE NUM " + str(ant_num))
                else:
                    #SUPPORT = (Num Transactions with Antecedent AND Consequent )/Total Num Transaction
                    #CONFIDENCE is likeliness of occurrence of Consequent Given the Antecedent
                    if ant_full == consequent:
                        new_row = [consequent, row[0], ant_base, ant_num, row[1],1,1]
                    else:
                        support = int(row[1])/int(total_num_transactions)
                        new_row = [consequent, row[0], ant_base, ant_num, row[1], support,1]
                    new_list.append(new_row)

        print(str(row[1]))


    with open(file_out_name2, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(new_list)


if __name__ == '__main__':
    main()
