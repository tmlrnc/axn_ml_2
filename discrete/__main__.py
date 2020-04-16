import argparse

from discrete.vl_kmeans_kmedian import K_Means, normalizer
from discrete.binize import VL_Binizer
from discrete.binize_kmeans import VL_Discretizer_KMeans
import csv

import pandas
import pandas as pd
import numpy

from ohe.encoder import OneHotEncoderBuilder
from discrete.discretizer import DiscretizerBuilder


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in')
    parser.add_argument('--file_out_ohe')
    parser.add_argument('--file_out_discrete')
    parser.add_argument('--file_out_ohe_dis')


    parser.add_argument('--file_out')
    parser.add_argument('--dicretize', nargs='+',  action='append')



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

    file_out_ohe_dis = args.file_out_ohe_dis
    vl_dicretize_list_many = args.dicretize

    ######################################################################
    #
    # Discretize
    #
    print("Discretize --- START ")
    file_in_name_org = file_in_name
    file_out_org = file_out



    #import random
    #for i in range(10000):
    #    print(str(random.random()) + ",")
    #exit()

    data_frame_all_len = pandas.read_csv(file_in_name_org).fillna(value=0)
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


