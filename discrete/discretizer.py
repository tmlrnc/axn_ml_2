import csv
import pandas
import numpy as np

from ohe.vl_kmeans_kmedian import K_Means, normalizer
from ohe.binize import VL_Binizer
from ohe.binize_kmeans import VL_Discretizer_KMeans

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

        for head in self.headers:
            nh = head + "DISCRETE"
            newhead.append(nh)

        with open(file_out_name, mode='a') as _file:
            _writer = csv.writer(_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if (self.write_header_flag == 1):
                _writer.writerow(newhead)
            for row in self.Xt_VL_K_list:
                _writer.writerow(row)



    def discretize_all(self):
        """

        :returns data_frame: array
        :returns csv_column_name_list: array

         """


        ############################# only in list


        print("discretize_list in discretize " + str(self.discretize_list))
        print("vl_discretize_list in discretize " + str(self.vl_discretize_list))


        #df = pd.read_csv(self.file_in_name, sep=',',usecols=["PPM"] )
        df = pd.read_csv(self.file_in_name, sep=',',usecols=self.discretize_list )

        X = df.to_numpy()

        my_strategy = self.discretize_strategy[0]
        print("my_strategy " + str(my_strategy))

        #bin = VL_Binizer(n_bins=5, encode='ordinal', strategy='uniform')

        bin = VL_Binizer(n_bins=5, encode='ordinal', strategy=my_strategy)

        bin.fit(X)
        Xt_VL = bin.transform(X)

        print("Xt_VL " + str(Xt_VL))
        print("bin_edges_ " + str(bin.bin_edges_))


        print('************************************* here')


        bin_AS = VL_Binizer(n_bins=5, encode='ordinal', strategy='analyst_supervised', edge_array=[.2, .5, .7])
        bin_AS.fit(X)
        Xt_VL = bin_AS.transform(X)

        print("analyst_supervised Xt_VL " + str(Xt_VL))
        print("analyst_supervised bin_edges_ " + str(bin_AS.bin_edges_))

        print('*************************************')
        bin_AS = VL_Binizer(n_bins=5, encode='ordinal', strategy='analyst_supervised', edge_array=[.3, .9])
        bin_AS.fit(X)
        Xt_VL = bin_AS.transform(X)

        print("analyst_supervised Xt_VL " + str(Xt_VL))
        print("analyst_supervised bin_edges_ " + str(bin_AS.bin_edges_))

        print('*************************************')

        bin_AS = VL_Binizer(n_bins=5, encode='ordinal', strategy='analyst_supervised', edge_array=[.5])
        bin_AS.fit(X)
        Xt_VL = bin_AS.transform(X)

        print("analyst_supervised Xt_VL " + str(Xt_VL))
        print("analyst_supervised bin_edges_ " + str(bin_AS.bin_edges_))

        print('*************************************')



        data_frame_all = pandas.read_csv(self.file_in_name,usecols=["PPM"])
        self.csv_column_name_list = list(data_frame_all.columns)
        X = data_frame_all.to_numpy()

        self.data_frame = data_frame_all
        bin_AS_K = VL_Discretizer_KMeans(n_bins=5, encode='ordinal', strategy='uniform')
        bin_AS_K.fit(X)
        Xt_VL_K = bin_AS_K.transform(X)

        print("analyst_supervised Xt_VL " + str(Xt_VL_K))
        print("analyst_supervised bin_edges_ " + str(bin_AS_K.bin_edges_))

        print('*************************************')

        df = pd.read_csv(self.file_in_name, sep=',', header=None)
        X = df.to_numpy()

        data_frame_all = pandas.read_csv(self.file_in_name,usecols=["PPM"])
        csv_column_name_list = list(data_frame_all.columns)
        X = data_frame_all.to_numpy()

        bin_AS_K = VL_Discretizer_KMeans(n_bins=5, encode='ordinal', strategy='analyst_supervised',
                                         edge_array=[.1, .5, .8])
        bin_AS_K.fit(X)
        Xt_VL_K = bin_AS_K.transform(X)

        print("analyst_supervised Xt_VL " + str(type(Xt_VL_K)))
        print("analyst_supervised bin_edges_ " + str(bin_AS_K.bin_edges_))

        self.headers = csv_column_name_list

        self.Xt_VL_K_list = list(Xt_VL_K)

        return self.data_frame, self.csv_column_name_list


    def discretize(self):
        """

        :returns data_frame: array
        :returns csv_column_name_list: array

         """

        data_frame_all = pd.read_csv(self.file_in_name, sep=',',usecols=self.discretize_list )
        self.data_frame = data_frame_all
        X = data_frame_all.to_numpy()
        self.csv_column_name_list = list(data_frame_all.columns)
        self.headers = self.csv_column_name_list
        my_strategy = self.discretize_strategy[0]
        my_bins = int(self.discretize_bins[0])

        print("my_strategy " + str(my_strategy))
        print("my_bins " + str(my_bins))

        if my_strategy == "uniform":

            binizer = VL_Binizer(n_bins=my_bins, encode='ordinal', strategy=my_strategy)
            binizer.fit(X)
            Xt_VL = binizer.transform(X)
            print("EDGES " + str(binizer.bin_edges_))
            print("TRANSFORMED COLUMN" + str(Xt_VL))
            self.Xt_VL_K_list = list(Xt_VL)
            print('UNIFORM strategy ************************************* ')

        elif my_strategy == "analyst_supervised":
            my_edge_array = []
            for ea in self.edge_array_list:
                my_edge_array.append(float(ea))

            binizer = VL_Discretizer_KMeans(n_bins=my_bins, encode='ordinal', strategy=my_strategy, edge_array=my_edge_array)
            binizer.fit(X)
            Xt_VL_K = binizer.transform(X)
            print("EDGES " + str(binizer.bin_edges_))
            print("TRANSFORMED COLUMN " + str(Xt_VL_K))
            self.Xt_VL_K_list = list(Xt_VL_K)
            print('analyst_supervised strategy ************************************* ')

        elif my_strategy == "kmeans":
            print('kmeans strategy ************************************* ')

            binizer = VL_Discretizer_KMeans(n_bins=my_bins, encode='ordinal', strategy=my_strategy)
            binizer.fit(X)
            Xt_VL_K = binizer.transform(X)
            print("EDGES " + str(binizer.bin_edges_))
            print("TRANSFORMED COLUMN " + str(Xt_VL_K))
            self.Xt_VL_K_list = list(Xt_VL_K)
            print('kmeans strategy ************************************* ')


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
