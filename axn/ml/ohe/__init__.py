"""
One Hot Encoder Module

<img src="images/ohe.png" alt="OHE">

Machine learning algorithms cannot work with categorical data directly.
Categorical data must be converted to numbers.
This module encodes categorical features as a one-hot numeric array.
A one hot encoding is a representation of categorical variables as binary vectors.
This first requires that the categorical values be mapped to integer values.
Then, each integer value is represented as a binary vector that is all zero values
except the index of the integer, which is marked with a 1.

The input to this module should be a csv file where each column feature is a category as a string like
"United States, Canada" or "Male, Female"

Then we take the string and convert to array of integers
denoting the values taken on by categorical (discrete) features.

Like
"United States, Canada" becomes (1,2)
"Male, Female" becomes (3,4)

The features are encoded using a one-hot aka ‘one-of-K’ or ‘dummy’ encoding scheme.
This creates a binary column for each category and returns a sparse matrix or dense array (depending on the sparse parameter)


Like
"United States, Canada" becomes (1,2) that encodes to (001, 010)
"Male, Female" becomes (3,4) that encodes to (100, 101)


By default, the encoder derives the categories based on the unique values in each feature.
Alternatively, you can also specify the categories manually.

This encoding is needed for feeding categorical data to many scikit-learn estimators, notably linear models and SVMs with the standard kernels.


Parameters
----------
file_in: file
    raw csv file input to be predicted. Must be a csv file where first row has column header
file_out: file
    csv file output encoded using one-hot one-of-K encoding scheme
ignore: string
    columns of data to NOT be encoded


Returns:
----------
    csv file output encoded using one-hot one-of-K encoding scheme.


Example 1. CSV Files:
---------------------
python -m ohe  \


  --file_in csvs/sales.csv \




  --file_out_ohe csvs/sales_ohe.csv \



  --ignore id


Example 1 - Data Input CSV File:
----------------------------
<img src="images/sales.png" alt="OHE" width="600" height="300">


Example 1 - One Hot Encoded CSV File:
-----------------------------
<img src="images/sales_ohe.png" alt="OHE" width="600" height="300">






Example 2:
----------
python -m ohe  \


  --file_in csvs/states.csv \




  --file_out_ohe csvs/states_ohe.csv \



  --ignore none



Example 2 Data Input File:
----------
ALASKA

ARIZONA

ARKANSAS

CALIFORNIA

COLORADO

CONNECTICUT

DELAWARE

FLORIDA

GEORGIA


Example 2 OHE Output File:
----------



1	0	0	0	0	0	0	0	0	0


0	1	0	0	0	0	0	0	0	0


0	0	1	0	0	0	0	0	0	0


0	0	0	1	0	0	0	0	0	0


0	0	0	0	1	0	0	0	0	0


0	0	0	0	0	1	0	0	0	0


0	0	0	0	0	0	1	0	0	0


0	0	0	0	0	0	0	1	0	0


0	0	0	0	0	0	0	0	1	0


0	0	0	0	0	0	0	0	0	1




Example 3:
----------
python -m ohe  \


  --file_in csvs/gender.csv \


  --file_out_ohe csvs/gender_ohe.csv \



  --ignore none




Example 3 Data Input File:
----------
MALE

MALE

FEMALE

FEMALE

MALE

FEMALE

FEMALE

MALE

FEMALE

FEMALE



Example 3 OHE Output File:
----------


0	1

0	1

1	0

1	0

0	1

1	0

1	0

0	1

1	0

1	0



Example 4:
----------
python -m ohe  \


  --file_in csvs/pets.csv \


  --file_out_ohe csvs/pets_ohe.csv \



  --ignore none


Example 4 Data Input File:
----------
DOG

CAT

CAT

NONE

DOG

DOG

NONE

NONE

CAT

DOG




Example 4 OHE Output File:
----------

0	1	0

1	0	0

1	0	0

0	0	1

0	1	0

0	1	0

0	0	1

0	0	1

1	0	0

0	1	0




"""


import argparse


from .encoder import OneHotEncoderBuilder


def parse_command_line():
    """
    reads the command line args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_in',
        help='raw csv file input to be predicted. Must be a csv file where first row has column header '
             'names. Must include time series date columns - like MM/DD/YY (7/3/20) ')
    parser.add_argument(
        '--file_out',
        help='csv file output encoded using one-hot one-of-K encoding scheme')
    parser.add_argument(
        '--ignore',
        action='append',
        help='columns of data to NOT be encoded or discrtizeed - remove from processing without '
             'removing from raw data because they might be usseful to know latrer - like first name')
    args = parser.parse_args()
    return args


def main():
    """
    return one hot encoded follwing these steps:


Step 1
----------
    READ FILE_IN_RAW.CSV


Step 2
----------
    GET COLUMN HEADERS



Step 3
----------
    FOR EACH COLUMN NOT IN IGNORE LIST




Step 4
----------
    GET ALL CATEGORIES = UNIQUE COLUMN VALUES



Step 5
----------
    GET COLUMN HEADERS



Step 6
----------
    ENCODE EACH ROW WITH 1 or 0 FOR EACH HEADER





      """
    ######################################################################
    #
    # read run commands
    #
    args = parse_command_line()
    file_in_name = args.file_in
    file_out = args.file_out

    ######################################################################

    #
    # OHE
    #
    print("OneHotEncoder --- START ")

    ohe_builder = OneHotEncoderBuilder(file_in_name)
    for ignore in args.ignore:
        ohe_builder.ignore(ignore)
    ohe = ohe_builder.build()
    data_frame, feature_name_list = ohe.one_hot_encode()

    print("data_frame " + str(data_frame))
    print("feature_name_list " + str(feature_name_list))
    ohe.write_ohe_csv(file_out)

    print("OneHotEncoder --- END ")
