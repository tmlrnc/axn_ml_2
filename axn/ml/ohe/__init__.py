"""

One Hot Encoder

 Step 1
----------
    READ FILE_IN_RAW.CSV
    reads config yaml file into config dictionary data object
    to drive the scikit learn machine learning algorithm


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
    OHE EACH ROW INTO 1,0

"""
# pylint: disable=invalid-name
# pylint: disable=import-error


import argparse


from axn.ml.ohe.encoder import OneHotEncoderBuilder


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
    parser.add_argument(
        '--ignore',
        action='append',
        help='columns of data to NOT be encoded or discrtizeed - remove from processing without '
             'removing from raw data because they might be usseful to know latrer - like first name')
    args = parser.parse_args()
    return args


def main():
    """
  READ FILE_IN_RAW.CSV
  GET COLUMN HEADERS
  FOR EACH COLUMN NOT IN IGNORE LIST :
  GET ALL CATEGORIES = UNIQUE COLUMN VALUES
  GENERATE ONE HOT ENCODING HEADER
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
