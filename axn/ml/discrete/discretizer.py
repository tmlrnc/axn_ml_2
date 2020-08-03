"""
Encode categorical features as a one-hot numeric array.

The input to this transformer should be an array-like of integers or strings,
denoting the values taken on by categorical (discrete) features.
The features are encoded using a one-hot one-of-K encoding scheme.
This creates a binary column for each category and returns a sparse matrix or dense array
depending on the sparse parameter the encoder derives the categories based on the unique values in each feature.
"""
# pylint: disable=import-error


import csv
import pandas as pd
from sklearn.cluster import DBSCAN
from axn.ml.discrete.binize import VlBinizer
from axn.ml.discrete.binize_kmeans import VlDiscretizerKmeans


class Discretizer():
    """
Encode categorical features as a one-hot numeric array.

The input to this transformer should be an array-like of integers or strings,
denoting the values taken on by categorical (discrete) features.
The features are encoded using a one-hot one-of-K encoding scheme.
This creates a binary column for each category and returns a sparse matrix or dense array
depending on the sparse parameter the encoder derives the categories based on the unique values in each feature.
    """
    # pylint: disable=too-many-instance-attributes
    # We are using this class to decide one of 8 discretize strategy paths
    def __init__(self, file_in, vl_discretize_list):
        """
        opens file and writes one hot encoded data

        :param ignore_list_in: list[] : list of feature to ignore
        :param file_in: string : input data file
        """
        self.file_in_name = file_in
        self.headers = []
        self.strategy = ""
        self.discretize_list = []
        self.discretize_strategy = []
        self.discretize_bins = []
        self.xt_vl_k_list = []
        self.vl_discretize_list = vl_discretize_list
        self.edge_array_list = []
        i = 0
        for dis in vl_discretize_list:
            if i == 0:
                self.discretize_strategy.append(dis)
            elif i == 1:
                self.discretize_bins.append(dis)
            elif i == 2:
                self.discretize_list.append(dis)
            elif i > 2:
                self.edge_array_list.append(dis)

            i = i + 1

        self.data_frame_all = pd.read_csv(file_in).fillna(value=0)
        len(self.data_frame_all)
        self.data_frame = self.data_frame_all
        self.data_frame = self.data_frame_all.drop(self.discretize_list, 1)
        self.data_frame_ignore_frame = self.data_frame_all[self.discretize_list]
        self.data_frame_ignore_frame_list = self.data_frame_ignore_frame.values.tolist()
        self.write_header_flag = 1
        self.csv_column_name_list = list(self.data_frame.columns)
        self.encoded = False


    def write_discretize_csv(self, file_out_name):
        """
        opens file and writes one hot encoded data

        :param file_out_name: Name of File to Write to
        """
        newhead = []

        drop = ""
        for head in self.headers:
            n_h = head + "_DISCRETE"
            drop = head
            print("head " + str(head))
            newhead.append(n_h)

        if self.strategy == "dbscan":

            print("dbscan")
            print(str(type(self.xt_vl_k_list)))

            with open(file_out_name, mode='a') as _file:
                _writer = csv.writer(
                    _file,
                    delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL)
                if self.write_header_flag == 1:
                    _writer.writerow(newhead)
                for row in self.xt_vl_k_list:
                    _writer.writerow([row])

        else:

            with open(file_out_name, mode='a') as _file:
                _writer = csv.writer(
                    _file,
                    delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL)
                if self.write_header_flag == 1:
                    _writer.writerow(newhead)
                for row in self.xt_vl_k_list:
                    _writer.writerow(row)
        return drop

    def discretize(self):
        """

        :returns data_frame: array
        :returns csv_column_name_list: array

         """

        df_0 = pd.read_csv(
            self.file_in_name,
            sep=',',
            usecols=self.discretize_list)
        data_frame_all = df_0.fillna(0)
        self.data_frame = data_frame_all
        x_my = data_frame_all.to_numpy()
        self.csv_column_name_list = list(data_frame_all.columns)
        self.headers = self.csv_column_name_list
        my_strategy = self.discretize_strategy[0]
        self.strategy = my_strategy
        my_bins = int(self.discretize_bins[0])

        if my_strategy == "uniform":

            binizer = VlBinizer(
                n_bins=my_bins,
                encode='ordinal',
                strategy=my_strategy)
            binizer.fit(x_my)

            xt_vl = binizer.transform(x_my)

            self.xt_vl_k_list = list(xt_vl)
            print('UNIFORM strategy ************************************* ')

        elif my_strategy == "analyst_supervised":
            my_edge_array = []
            for e_a in self.edge_array_list:
                my_edge_array.append(float(e_a))

            binizer = VlDiscretizerKmeans(
                n_bins=my_bins,
                encode='ordinal',
                strategy=my_strategy,
                edge_array=my_edge_array)
            binizer.fit(x_my)
            xt_vl_k = binizer.transform(x_my)

            self.xt_vl_k_list = list(xt_vl_k)
            print('analyst_supervised strategy ************************************* ')

        elif my_strategy == "kmeans":
            print('kmeans strategy ************************************* ')

            binizer = VlDiscretizerKmeans(
                n_bins=my_bins, encode='ordinal', strategy=my_strategy)
            binizer.fit(x_my)
            xt_vl_k = binizer.transform(x_my)

            self.xt_vl_k_list = list(xt_vl_k)

            print('kmeans strategy ************************************* ')

        elif my_strategy == "dbscan":
            print('dbscan strategy ************************************* ')
            print(str(type(x_my)))

            clustering = DBSCAN(eps=my_bins / 10, min_samples=1).fit(x_my)

            self.xt_vl_k_list = list(clustering.labels_.tolist())

            print('dbscan strategy ************************************* ')

        return self.data_frame, self.csv_column_name_list


class DiscretizerBuilder():
    """
    Builder Design Pattern for Discretizer Class

    """
    def __init__(self, filename):
        """
        opens file and writes one hot encoded data

        :param filename: string : input data file
        """
        if filename is None:
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
