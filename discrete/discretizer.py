import csv
import pandas
import numpy as np

from discrete.vl_kmeans_kmedian import K_Means, normalizer
from discrete.binize import VL_Binizer
from discrete.binize_kmeans import VL_Discretizer_KMeans

import pandas
import pandas as pd
import numpy

class Discretizer(object):
    """

    """
    def __init__(self, file_in,vl_discretize_list):
        """
        opens file and writes one hot encoded data

        :param ignore_list_in: list[] : list of feature to ignore
        :param file_in: string : input data file
        """
        self.file_in_name = file_in
        self.discretize_list = []
        self.discretize_strategy = []
        self.discretize_bins = []
        self.Xt_VL_K_list = []
        self.vl_discretize_list = vl_discretize_list
        self.edge_array_list = []
        i = 0
        for dis in vl_discretize_list:
            if i == 0 :
                self.discretize_strategy.append(dis)
            elif i == 1:
                self.discretize_bins.append(dis)
            elif i == 2 :
                self.discretize_list.append(dis)
            elif i > 2:
                self.edge_array_list.append(dis)

            i = i + 1

        self.data_frame_all = pandas.read_csv(file_in).fillna(value = 0)
        len(self.data_frame_all)
        self.data_frame = self.data_frame_all
        self.data_frame = self.data_frame_all.drop(self.discretize_list, 1)
        self.data_frame_ignore_frame = self.data_frame_all[self.discretize_list]
        self.data_frame_ignore_frame_list = self.data_frame_ignore_frame.values.tolist()
        self.write_header_flag = 1
        self.csv_column_name_list = list(self.data_frame.columns)
        self.encoded = False

        return

    def write_discretize_csv(self,file_out_name):
        """
        opens file and writes one hot encoded data

        :param file_out_name: Name of File to Write to
        """
        newhead = []

        drop = ""
        for head in self.headers:
            nh = head + "_DISCRETE"
            drop = head
            print("head " + str(head))
            newhead.append(nh)


        if(self.ss == "dbscan") :

            print("dbscan")
            print(str(type(self.Xt_VL_K_list)))

            with open(file_out_name, mode='a') as _file:
                _writer = csv.writer(_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if (self.write_header_flag == 1):
                    _writer.writerow(newhead)
                for row in self.Xt_VL_K_list:
                    _writer.writerow([row])


        else:

            with open(file_out_name, mode='a') as _file:
                _writer = csv.writer(_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if (self.write_header_flag == 1):
                    _writer.writerow(newhead)
                for row in self.Xt_VL_K_list:
                    _writer.writerow(row)
        return drop




    def discretize(self):
        """

        :returns data_frame: array
        :returns csv_column_name_list: array

         """

        df_0 = pd.read_csv(self.file_in_name, sep=',',usecols=self.discretize_list )
        data_frame_all = df_0.fillna(0)
        self.data_frame = data_frame_all
        X = data_frame_all.to_numpy()
        self.csv_column_name_list = list(data_frame_all.columns)
        self.headers = self.csv_column_name_list
        my_strategy = self.discretize_strategy[0]
        self.ss = my_strategy
        my_bins = int(self.discretize_bins[0])


        if my_strategy == "uniform":



            binizer = VL_Binizer(n_bins=my_bins, encode='ordinal', strategy=my_strategy)
            binizer.fit(X)


            Xt_VL = binizer.transform(X)

            self.Xt_VL_K_list = list(Xt_VL)
            print('UNIFORM strategy ************************************* ')



        elif my_strategy == "analyst_supervised":
            my_edge_array = []
            for ea in self.edge_array_list:
                my_edge_array.append(float(ea))

            binizer = VL_Discretizer_KMeans(n_bins=my_bins, encode='ordinal', strategy=my_strategy, edge_array=my_edge_array)
            binizer.fit(X)
            Xt_VL_K = binizer.transform(X)

            self.Xt_VL_K_list = list(Xt_VL_K)
            print('analyst_supervised strategy ************************************* ')

        elif my_strategy == "kmeans":
            print('kmeans strategy ************************************* ')

            binizer = VL_Discretizer_KMeans(n_bins=my_bins, encode='ordinal', strategy=my_strategy)
            binizer.fit(X)
            Xt_VL_K = binizer.transform(X)

            self.Xt_VL_K_list = list(Xt_VL_K)


            print('kmeans strategy ************************************* ')


        elif my_strategy == "dbscan":
            print('dbscan strategy ************************************* ')
            from sklearn.cluster import DBSCAN
            print(str(type(X)))

            clustering = DBSCAN(eps=my_bins/10, min_samples=1).fit(X)


            self.Xt_VL_K_list = list(clustering.labels_.tolist())



            print('dbscan strategy ************************************* ')


        return self.data_frame, self.csv_column_name_list




class DiscretizerBuilder(object):
    def __init__(self, filename):
        """
        opens file and writes one hot encoded data

        :param filename: string : input data file
        """
        if filename == None:
            raise Exception("Filename cannot be none")
        self.filename = filename
        self.vl_discretize_list = []

    def discretize(self, discretize_list):
        """
        constructs ignore list

        :param ignore: string : one feature string on list
        """
        self.vl_discretize_list = discretize_list
        return self

    def build(self):
        """
        builds OHE class

        :returns DiscretizerB: class

        """

        return Discretizer(self.filename, self.vl_discretize_list)
