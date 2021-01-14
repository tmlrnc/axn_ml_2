
from subprocess import call

#!/usr/bin/env python3
# Counselors19*

import sys
import argparse
import csv
print("start")

def lists_are_same(tsnl,csl):
    i = 0
    mytest = True
    for ele in tsnl:
        print(csl[i])
        print(tsnl[i])
        if tsnl[i] != csl[i] :
            mytest = False
            print("False")
    return mytest

def get_dataset(f):
    return set(map(tuple, csv.reader(f)))


def do_diff(f1, f2):
    set1 = get_dataset(f1)
    set2 = get_dataset(f2)
    different = set1 ^ set2
    empty = set()
    return empty == different

ROOT_NAME = "/Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/"
TEST_OUTPUT_FILE = ROOT_NAME + "MBA_IN_TEST1_OUT.csv"
EXPECTED_OUTPUT_FILE = ROOT_NAME + "MBA_IN_TEST1_OUT_EO.csv"

def my_file_test(f1, f2):
    thediff = do_diff(f1,f2)
    print(thediff)
    return  thediff

def test_answer1():
    rc = call("./mba_test_1.sh")
    assert my_file_test(TEST_OUTPUT_FILE,EXPECTED_OUTPUT_FILE) == True

TEST_OUTPUT_FILE2 = ROOT_NAME + "MBA_IN_TEST2_OUT.csv"
EXPECTED_OUTPUT_FILE2 = ROOT_NAME + "MBA_IN_TEST2_OUT_EO.csv"

def test_answer2():
    rc = call("./mba_test_2.sh")
    assert my_file_test(TEST_OUTPUT_FILE2,EXPECTED_OUTPUT_FILE2) == True


TEST_OUTPUT_FILE3 = ROOT_NAME + "MBA_IN_TEST3_OUT.csv"
EXPECTED_OUTPUT_FILE3 = ROOT_NAME + "MBA_IN_TEST3_OUT_EO.csv"
def test_answer3():
    rc = call("./mba_test_3.sh")
    assert my_file_test(TEST_OUTPUT_FILE3,EXPECTED_OUTPUT_FILE3) == True

from . import mba

def test_answer4():
    import pandas as pd
    test_data_in = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0]
    ]
    test_df_in = pd.DataFrame(test_data_in, columns=['Married_M', 'Married_S', 'HealthWell_0.0', 'HealthWell_1.0'])

    calc_df_out = mba(test_df_in)
    calc_sup = calc_df_out['support'].round(1)
    calc_conf = calc_df_out['confidence'].round(1)

    data_out = [
        ['Married_M', 'HealthWell_0.0',0.222222,0.400000,0.7200],
        ['Married_S', 'HealthWell_0.0',0.333333 , 0.75, 1.3500],
        ['Married_M', 'HealthWell_1.0',0.333333,0.600000,1.3500],
        ['Married_M', 'HealthWell_1.0', 0.111111, 0.25, 0.5625],
        ['HealthWell_0.0', 'Married_M', 0.222222, 0.40, 0.7200],
        ['HealthWell_1.0', 'Married_M',0.333333, 0.75, 1.3500],
        ['HealthWell_0.0', 'Married_S', 0.333333, 0.60,1.35],
        ['HealthWell_1.0', 'Married_S', 0.111111, 0.25, 0.5625]

    ]
    test_df_out = pd.DataFrame(data_out, columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    test_sup = test_df_out['support'].round(1)
    test_conf = test_df_out['confidence'].round(1)
    tsn = test_sup.rename('support')

    tsnl = tsn.tolist()
    csl = calc_sup.tolist()

    new_test = lists_are_same(csl,tsnl)
    print(new_test)

    assert new_test == True


def test_answer5():
    import pandas as pd
    test_data_in = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0]
    ]
    test_df_in = pd.DataFrame(test_data_in, columns=['Jewlry_NO', 'Jewlry_YES', 'HOUSE_YES', 'HOUSE_NO'])

    calc_df_out = mba(test_df_in)
    print("*********")

    print(calc_df_out)

    calc_sup = calc_df_out['support'].round(1)
    calc_conf = calc_df_out['confidence'].round(1)

    data_out = [
        ['Jewlry_NO', 'HOUSE_NO.0',0.333333,0.571429,1.371429],
        ['Jewlry_YES', 'HOUSE_NO',0.333333 , 0.200000, 0.480000],
        ['Jewlry_NO', 'HOUSE_YES.0',0.333333,0.428571,0.734694],
        ['Jewlry_YES', 'HOUSE_YES', 0.111111,  0.800000, 1.371429],
        ['HOUSE_NO', 'Jewlry_NO', 0.222222, 0.800000, 1.371429],
        ['HOUSE_YES', 'Jewlry_NO',0.333333, 0.428571, 0.734694],
        ['HOUSE_NO', 'Jewlry_YES', 0.333333, 0.200000,0.480000],
        ['HOUSE_YES', 'Jewlry_YES', 0.111111, 0.571429, 1.371429]

    ]
    test_df_out = pd.DataFrame(data_out, columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    test_conf = test_df_out['confidence'].round(1)
    tsn = test_conf.rename('support')

    tsnl = tsn.tolist()
    csl = calc_conf.tolist()

    new_test = lists_are_same(csl,tsnl)
    print(new_test)

    assert new_test == True

def test_answer6():
    import pandas as pd
    test_data_in = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0]
    ]
    test_df_in = pd.DataFrame(test_data_in, columns=['Jewlry_NO', 'Jewlry_YES', 'HOUSE_YES', 'HOUSE_NO'])

    calc_df_out = mba(test_df_in)
    print("*********")

    print(calc_df_out)

    calc_sup = calc_df_out['support'].round(1)
    calc_conf = calc_df_out['confidence'].round(1)
    calc_lift = calc_df_out['lift'].round(1)


    data_out = [
        ['Jewlry_NO', 'HOUSE_NO.0',0.333333,0.571429,1.371429],
        ['Jewlry_YES', 'HOUSE_NO',0.333333 , 0.200000, 0.480000],
        ['Jewlry_NO', 'HOUSE_YES.0',0.333333,0.428571,0.734694],
        ['Jewlry_YES', 'HOUSE_YES', 0.111111,  0.800000, 1.371429],
        ['HOUSE_NO', 'Jewlry_NO', 0.222222, 0.800000, 1.371429],
        ['HOUSE_YES', 'Jewlry_NO',0.333333, 0.428571, 0.734694],
        ['HOUSE_NO', 'Jewlry_YES', 0.333333, 0.200000,0.480000],
        ['HOUSE_YES', 'Jewlry_YES', 0.111111, 0.571429, 1.371429]

    ]
    test_df_out = pd.DataFrame(data_out, columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    test_conf = test_df_out['lift'].round(1)
    tsn = test_conf.rename('lift')

    tsnl = tsn.tolist()
    csl = calc_lift.tolist()

    new_test = lists_are_same(csl,tsnl)
    print(new_test)

    assert new_test == True