import argparse

from discrete.vl_kmeans_kmedian import K_Means, normalizer
from discrete.binize import VL_Binizer
from discrete.binize_kmeans import VL_Discretizer_KMeans
import csv

import pandas
import pandas as pd
import numpy
import numpy as np


from ohe.encoder import OneHotEncoderBuilder
from discrete.discretizer import DiscretizerBuilder

description = \
"""
VoterLabs Inc. Data Discretizer

READ FILE_IN_RAW.CSV
GET COLUMN HEADERS
FOR EACH COLUMN NOT IN IGNORE LIST :
GET ALL CATEGORIES = UNIQUE COLUMN VALUES
GENERATE ONE HOT ENCODING HEADER
ENCODE EACH ROW WITH 1 or 0 FOR EACH HEADER

Standardization discretization and one hot encoding is an important requirement for VL machine learning estimators.
VL models WILL behave badly if the individual features do not look like standard normally distributed data: 
that is, Gaussian with zero mean and unit variance. VL ignored the distribution and just transforms the data 
to center it by removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.
This is done by transforming conntious data into bins or one hot encoding 
      """.strip()


def parse_command_line():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--file_in', help='raw csv file input to be discretized')
    parser.add_argument('--file_out_ohe', help='file intermediate  .')
    parser.add_argument('--file_out_discrete', help='file intermediate')
    parser.add_argument('--file_out_ohe_dis', help='file intermediate')
    parser.add_argument(
        '--drop_column',
        action='append', help='drop column from discreteize process - BUT not from encoding or prediction')
    parser.add_argument('--ignore',
        action = 'append', help = 'columns of data to NOT be encoded or discretized - remove from processing without removing from raw data because they might be usseful to know latrer - like first name')

    parser.add_argument('--file_out', help='output file of discretize process where continous data in transformed into bins used for models ')
    parser.add_argument('--dicretize', nargs='+',  action='append', help='discretization strategy - uniform, quantile analyst_supervised kmeans')



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
    file_out_discrete = args.file_out_discrete
    file_out = args.file_out
    ignore = args.ignore

    file_out_ohe_dis = args.file_out_ohe_dis
    vl_dicretize_list_many = args.dicretize

    ######################################################################
    #
    # Discretize
    #
    print("Discretize --- START ")

    drop_column = args.drop_column




    file_in_name_org = file_in_name
    file_in_name_drop = file_in_name

    dropname = "_drop.csv"
    file_in_name_drop = file_in_name.replace(".csv", dropname)
    dfd = pandas.read_csv(file_in_name_org).fillna(value=0)
    dfd2 = dfd.drop(drop_column, axis=1)
    dfd2.to_csv(file_in_name_drop,index=False)

    file_in_name = file_in_name_drop
    file_out_org = file_out
    data_frame_all_len = pandas.read_csv(file_in_name_drop).fillna(value=0)
    mycol = data_frame_all_len.columns

    my_len = len(data_frame_all_len)
    i = 0
    for vl_dicretize_list_one in vl_dicretize_list_many:
        discretizer_builder = DiscretizerBuilder(file_in_name)
        discretizer_builder.discretize(vl_dicretize_list_one)
        discretizer = discretizer_builder.build()
        discretizer.discretize()
        new_end = str(i) + ".csv"
        new_file = file_out_discrete.replace(".csv", new_end)
        drop = discretizer.write_discretize_csv(new_file)
        print("drop " + str(drop))
        discrete_out_df = pd.read_csv(new_file)
        ohe_out_df = pd.read_csv(file_in_name)
        df_dis_ohe_result = discrete_out_df.join(ohe_out_df)
        dl = [drop]
        dll = df_dis_ohe_result.drop(dl, axis=1)
        new_end_out = str(i) + ".csv"
        new_file_out = file_out.replace(".csv", new_end_out)
        dll.to_csv(new_file_out, index=False)
        file_in_name  = new_file_out
        i = i + 1

    dll2 = dll[:my_len]
    dll2.to_csv(file_out_org,index=False)
    print("Discretize --- END ")


def _main():
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
    file_out_discrete = args.file_out_discrete
    file_out = args.file_out
    ignore = args.ignore

    file_out_ohe_dis = args.file_out_ohe_dis
    vl_dicretize_list_many = args.dicretize

    ######################################################################
    #
    # Discretize
    #
    print("Discretize --- START ")

    drop_column = args.drop_column




    file_in_name_org = file_in_name
    file_in_name_drop = file_in_name

    dropname = "_drop.csv"
    file_in_name_drop = file_in_name.replace(".csv", dropname)
    dfd = pandas.read_csv(file_in_name_org).fillna(value=0)
    dfd2 = dfd.drop(drop_column, axis=1)
    dfd2.to_csv(file_in_name_drop,index=False)

    file_in_name = file_in_name_drop

    file_out_org = file_out
    data_frame_all_len = pandas.read_csv(file_in_name_drop).fillna(value=0)

    mycol = data_frame_all_len.columns

    my_len = len(data_frame_all_len)
    i = 0
    for vl_dicretize_list_one in vl_dicretize_list_many:
        discretizer_builder = DiscretizerBuilder(file_in_name)
        discretizer_builder.discretize(vl_dicretize_list_one)
        discretizer = discretizer_builder.build()
        discretizer.discretize()
        new_end = str(i) + ".csv"
        new_file = file_out_discrete.replace(".csv", new_end)
        drop = discretizer.write_discretize_csv(new_file)
        print("drop " + str(drop))
        discrete_out_df = pd.read_csv(new_file)
        ohe_out_df = pd.read_csv(file_in_name)
        df_dis_ohe_result = discrete_out_df.join(ohe_out_df)
        dl = [drop]
        dll = df_dis_ohe_result.drop(dl, axis=1)
        new_end_out = str(i) + ".csv"
        new_file_out = file_out.replace(".csv", new_end_out)
        dll.to_csv(new_file_out, index=False)
        file_in_name  = new_file_out
        i = i + 1

    dll2 = dll[:my_len]
    dll2.to_csv(file_out_org,index=False)
    print("Discretize --- END ")

def __main():
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
    file_out_discrete = args.file_out_discrete
    file_out = args.file_out
    ignore = args.ignore

    file_out_ohe_dis = args.file_out_ohe_dis
    vl_dicretize_list_many = args.dicretize

    ######################################################################
    #
    # Discretize
    #
    print("Discretize --- START ")

    drop_column = args.drop_column

    ignore_list = []
    for ig in ignore:
        ignore_list.append(ig)

    file_in_name_org = file_in_name
    file_in_name_drop = file_in_name

    dropname = "_drop.csv"
    file_in_name_drop = file_in_name.replace(".csv", dropname)
    dfd = pandas.read_csv(file_in_name_org).fillna(value=0)
    dfd2 = dfd.drop(drop_column, axis=1)
    dfd2.to_csv(file_in_name_drop,index=False)

    file_in_name = file_in_name_drop

    file_out_org = file_out
    data_frame_all_len = pandas.read_csv(file_in_name_drop).fillna(value=0)



    data_frame_all = pandas.read_csv(file_in_name).fillna(value=0)

    data_frame_org = data_frame_all

    data_frame_org = data_frame_all.drop(ignore_list, 1)

    data_frame_ignore_frame = data_frame_all[ignore_list]
    # print("self.data_frame_ignore_frame " + str(self.data_frame_ignore_frame))

    print("self.data_frame_ignore_frame ")

    print(data_frame_ignore_frame)

    file_in_name = data_frame_org


    mycol = data_frame_all_len.columns

    my_len = len(data_frame_all_len)
    i = 0
    for vl_dicretize_list_one in vl_dicretize_list_many:
        discretizer_builder = DiscretizerBuilder(file_in_name)
        discretizer_builder.discretize(vl_dicretize_list_one)
        discretizer = discretizer_builder.build()
        discretizer.discretize()
        new_end = str(i) + ".csv"
        new_file = file_out_discrete.replace(".csv", new_end)
        drop = discretizer.write_discretize_csv(new_file)
        print("drop " + str(drop))
        discrete_out_df = pd.read_csv(new_file)
        ohe_out_df = pd.read_csv(file_in_name)
        df_dis_ohe_result = discrete_out_df.join(ohe_out_df)
        dl = [drop]
        dll = df_dis_ohe_result.drop(dl, axis=1)
        new_end_out = str(i) + ".csv"
        new_file_out = file_out.replace(".csv", new_end_out)
        dll.to_csv(new_file_out, index=False)
        file_in_name  = new_file_out
        i = i + 1

    dll2 = dll[:my_len]
    dll2.to_csv(file_out_org,index=False)
    print("Discretize --- END ")



if __name__ == '__main__':
    main()


