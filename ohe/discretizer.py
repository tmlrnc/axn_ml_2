import csv
import pandas
import numpy as np

from ohe.vl_kmeans_kmedian import K_Means, normalizer
from sklearn.preprocessing import KBinsDiscretizer
from ohe.binize import VL_Binizer
from ohe.binize_kmeans import VL_Discretizer_KMeans

import pandas
import pandas as pd
import numpy

class Discretizer(object):
    """
    features are encoded using a one-hot ‘one-of-K’ encoding scheme.
    This creates a binary column for each category and returns a sparse matrix or dense array
    the encoder derives the categories based on the unique values in each feature.

     when features are categorical.
     For example a person could have features
     ["male", "female"],
     ["from Europe", "from US", "from Asia"],
     ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"].
     Such features can be efficiently coded as integers,
     for instance ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3]
     while ["female", "from Asia", "uses Chrome"] would be [1, 2, 1].

    READ FILE_IN_RAW.CSV
    GET COLUMN HEADERS
    FOR EACH COLUMN NOT IN IGNORE LIST :
    GET ALL CATEGORIES = UNIQUE COLUMN VALUES
    GENERATE ONE HOT ENCODING HEADER
    ENCODE EACH ROW WITH 1 or 0 FOR EACH HEADER
    """
    def __init__(self, file_in,discretize_list):
        """
        opens file and writes one hot encoded data

        :param ignore_list_in: list[] : list of feature to ignore
        :param file_in: string : input data file
        """
        self.file_in_name = file_in
        self.discretize_list = discretize_list

        self.data_frame_all = pandas.read_csv(file_in).fillna(value = 0)

        self.data_frame = self.data_frame_all

        self.data_frame = self.data_frame_all.drop(self.discretize_list, 1)


        self.data_frame_ignore_frame = self.data_frame_all[self.discretize_list]
        #print("self.data_frame_ignore_frame " + str(self.data_frame_ignore_frame))

        #print("self.data_frame_ignore_frame t ")

        print(type(self.data_frame_ignore_frame))

        self.data_frame_ignore_frame_list = self.data_frame_ignore_frame.values.tolist()

        #print("self.data_frame_ignore_frame_list " + str(self.data_frame_ignore_frame_list))

        self.write_header_flag = 1
        self.csv_column_name_list = list(self.data_frame.columns)
        self.encoded = False

        return

    def write_discretize_csv(self,file_out_name):
        """
        opens file and writes one hot encoded data

        :param file_out_name: Name of File to Write to
        """
        #self.headers = self.headers + "DISCTETE"
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


    def discretize(self):
        """

        :returns data_frame: array
        :returns csv_column_name_list: array

         """


        ############################# only in list


        df = pd.read_csv(self.file_in_name, sep=',',usecols=["PPM"] )
        X = df.to_numpy()



        bin = VL_Binizer(n_bins=5, encode='ordinal', strategy='uniform')
        bin.fit(X)
        Xt_VL = bin.transform(X)

        print("Xt_VL " + str(Xt_VL))
        print("bin_edges_ " + str(bin.bin_edges_))

        print('*************************************')


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




class DiscretizerBuilder(object):
    def __init__(self, filename):
        """
        opens file and writes one hot encoded data

        :param filename: string : input data file
        """
        if filename == None:
            raise Exception("Filename cannot be none")
        self.filename = filename
        self.discretize_list = []

    def discretize(self, discretize):
        """
        constructs ignore list

        :param ignore: string : one feature string on list
        """
        self.discretize_list.append(discretize)
        return self

    def build(self):
        """
        builds OHE class

        :returns OneHotEncoder: class

        """
        return Discretizer(self.filename, self.discretize_list)
