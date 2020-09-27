"""

Market Basket Analysis REPORT for Association Rules
Support ( A for C)  = Number of Transaction with BOTH Antecedent AND Consequent ONLY / Total Number of ALL Transactions

SUPPORT is how frequent an Antecedent is in all the transactions
SUPPORT = (Num Transactions with Antecedent AND Consequent )/Total Num Transaction

CONFIDENCE is likeliness of occurrence of Consequent Given the Antecedent
CONFIDENCE = (Num Transactions with Antecedent AND Consequent )/ Num Transactions with Antecedent

Lift is how much better a Antecedent is at predicting the Consequent than just assuming the Consequent in the first place.
Lift = ((Num Transactions with Antecedent AND Consequent )/ Num Transactions with Antecedent) )
/ ((Num Transactions with Consequent) / Total Num Transaction)


Step 1
----------

Zero - all rows that have blanks

python -m zeroblank --file_in /Users/tomlorenc/Downloads/9_27_MBA_INPUT_V3.csv --file_out /Users/tomlorenc/Downloads/9_27_MBA_INPUT_V3_zero.csv



Step 2
----------


ADD ID COLUMN
One hot encode - all strings and integers to categories

Python -m ohe --file_in /Users/tomlorenc/Downloads/9_27_MBA_INPUT_V2_zero.csv --file_out /Users/tomlorenc/Downloads/9_27_MBA_INPUT_V2_zero_ohe.csv --ignore ID



Step 3
----------



REMOVE ID COLUMN
Count

Python -m count --file_in /Users/tomlorenc/Downloads/9_27_MBA_INPUT_V2_zero_ohe.csv --file_out /Users/tomlorenc/Downloads/9_27_MBA_INPUT_V2_zero_ohe_count.csv


Step 4
----------



Count Report


Python -m report --file_in /Users/tomlorenc/Downloads/9_27_MBA_INPUT_V2_zero_ohe_count.csv       --col_in /Users/tomlorenc/Downloads/9_27_MBA_INPUT_V2.csv --file_out /Users/tomlorenc/Downloads/9_27_MBA_INPUT_V2_zero_ohe_count_report.csv



Step 5
----------


REMOVE ID COLUMN
MBA


Python -m market_basket_analysis --file_in /Users/tomlorenc/Downloads/Y_VALUES_V3_zero_ohe.csv --file_out /Users/tomlorenc/Downloads/Y_VALUES_V3_zero_ohe_results.csv


MBA Report


Python -m mba_report --file_in /Users/tomlorenc/Downloads/Y_VALUES_V3_zero_ohe_results.csv --col_in /Users/tomlorenc/Downloads/Y_VALUES_V3.csv --file_out /Users/tomlorenc/Downloads/Y_VALUES_V3_zero_ohe_results_report.csv


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
        '--count_in',
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
    count_in = args.count_in

    file_in_name2 = args.file_in
    file_out_name2 = args.file_out

    col_in_df = pd.read_csv(col_in)

    import numpy as np

    mycols = col_in_df.columns

    num_of_rows = len(col_in_df)



    count_in_df = pd.read_csv(count_in)

    mycols_count = count_in_df.columns

    allrows = []
    for col in mycols_count:
        print(col)
        col_count = np.count_nonzero(count_in_df[col])
        my_row = [col, col_count]
        allrows.append(my_row)

    import csv

    with open(file_in_name2, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)


    total_num_transactions = 0
    new_list = []
    first_row = ["Consequent", "Antecedent",  "Support", "Confidence", "Lift" , "Consequent_Count"]
    #first_row = ["", "", "", "", "", "how frequent an Antecedent is", "likeliness of Consequent Given the Antecedent"]

    #new_list.append(first_row)

    for row in data:
        print(str(row[0]))
        ant_full = str(row[0])
        ant_full_2 = str(row[1])

        if (ant_full.find(',') != -1):
            print("comma NUM " + str(ant_full))
        #else:
        elif (ant_full_2.find('Y') != -1 ):
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
                    Consequent_Count = 0
                    if ant_num == "0":
                        print("ERASE NUM " + str(ant_num))
                    else:
                        Consequent_Count = 0

                        for cc_row in allrows:
                            if cc_row[0] == str(row[0]):
                                Consequent_Count = cc_row[1]
                        new_row = [row[0], row[1], row[2],row[3], row[4], Consequent_Count]
                        print("******************************************** " + str(new_row))

                        new_list.append(new_row)

            print(str(row[1]))

    #mydf = df = pd.DataFrame(new_list)

    mydf = df = pd.DataFrame(new_list,columns=first_row)
    mydf.sort_values(by=['Consequent'])
    mydf.to_csv(file_out_name2)



if __name__ == '__main__':
    main()
