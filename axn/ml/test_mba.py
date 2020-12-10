# content of test_sample.py

from subprocess import call

#!/usr/bin/env python3
# Counselors19*

import sys
import argparse
import csv
print("start")


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