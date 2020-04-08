import argparse

from ohe.config import init_ohe_config
from ohe.vl_kmeans_kmedian import K_Means, normalizer
from ohe.binize import VL_Binizer
from ohe.binize_kmeans import VL_Discretizer_KMeans
import csv

import pandas
import pandas as pd
import numpy


from ohe.discretizer import DiscretizerBuilder


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in')
    parser.add_argument('--file_out_discrete')
    parser.add_argument('--dicretize', nargs='+')
    args = parser.parse_args()
    return args

def main():
    """
    """
    ######################################################################
    #
    # read run commands
    #
    args = parse_command_line()
    file_in_name = args.file_in
    file_out_discrete = args.file_out_discrete
    vl_dicretize_list = args.dicretize

    ######################################################################
    #
    # Discretize
    #
    discretizer_builder = DiscretizerBuilder(file_in_name)
    discretizer_builder.discretize(vl_dicretize_list)
    discretizer = discretizer_builder.build()
    discretizer.discretize()
    discretizer.write_discretize_csv(file_out_discrete)


if __name__ == '__main__':
    main()


