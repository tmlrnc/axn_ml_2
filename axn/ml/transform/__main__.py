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

    df2['axn_Ppl_MarriedPrtner'] = df2['axn_Ppl_MarriedPrtner'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_ArtsCul'] = df2['axn_CnsmBhvr_ArtsCul'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_FoodOrgNat'] = df2['axn_CnsmBhvr_FoodOrgNat'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_HealthWell'] = df2['axn_CnsmBhvr_HealthWell'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_Crafts'] = df2['axn_CnsmBhvr_Crafts'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_DIY'] = df2['axn_CnsmBhvr_DIY'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_HmDecor'] = df2['axn_CnsmBhvr_HmDecor'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_Lwngrd'] = df2['axn_CnsmBhvr_Lwngrd'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_NewsJunkie'] = df2['axn_CnsmBhvr_NewsJunkie'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_SprtsFan'] = df2['axn_CnsmBhvr_SprtsFan'].apply(lambda x: 0 if x < 6 else 1)



    df2['axn_HH_YngKids'] = df2['axn_HH_YngKids'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_HH_TeensPreteens'] = df2['axn_HH_TeensPreteens'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_HH_GrndKids'] = df2['axn_HH_GrndKids'].apply(lambda x: 0 if x < 6 else 1)
    df2['JewelryCoverqage'] = df2['JewelryCoverqage'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_HH_Millennial'] = df2['axn_HH_Millennial'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_HH_Pets'] = df2['axn_HH_Pets'].apply(lambda x: 0 if x < 6 else 1)



    df2['axn_CnsmBhvr_AthFitness'] = df2['axn_CnsmBhvr_AthFitness'].apply(lambda x: 0 if x < 6 else 1)

    df2['axn_CnsmBhvr_FoodWine'] = df2['axn_CnsmBhvr_FoodWine'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_Gamcasno'] = df2['axn_CnsmBhvr_Gamcasno'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_Mtrcycl'] = df2['axn_CnsmBhvr_Mtrcycl'].apply(lambda x: 0 if x < 6 else 1)


    df2['axn_CnsmBhvr_Music'] = df2['axn_CnsmBhvr_Music'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_Outdoors'] = df2['axn_CnsmBhvr_Outdoors'].apply(lambda x: 0 if x < 6 else 1)

    df2['axn_CnsmBhvr_Sportsmn'] = df2['axn_CnsmBhvr_Sportsmn'].apply(lambda x: 0 if x < 6 else 1)
    df2['axn_CnsmBhvr_Travel'] = df2['axn_CnsmBhvr_Travel'].apply(lambda x: 0 if x < 6 else 1)

    df2.to_csv(file_out_name2)

    print("MBA --- END ")




if __name__ == '__main__':
    main()
