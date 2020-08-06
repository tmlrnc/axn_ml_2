"""
One Hot Encoder Module


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


Example 1:
----------
python -m ohe  \


  --file_in csvs/C102_PLUS_D.csv \


  --file_out_ohe csvs/C102_PLUS_D_OHE.csv \



  --file_out_predict csvs/C102_PLUS_D_OHE_PREDICT.csv \



  --file_in_config config/ohe_config_RUN1.yaml \



  --ignore deaths_DISCRETE \



  --ignore deaths \



  --target STATUS




Example 2:
----------
python -m ohe  \


  --file_in csvs/covid.csv \


  --file_out_ohe csvs/covid_ohe.csv \



  --ignore deaths_DISCRETE \



  --ignore deaths








"""

import csv
from sklearn.preprocessing import OneHotEncoder
import pandas
import numpy as np


class VLOneHotEncoder():
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

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=no-value-for-parameter
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, file_in, ignore_list_in):
        """
        opens file and writes one hot encoded data

        :param ignore_list_in: list[] : list of feature to ignore
        :param file_in: string : input data file
        """
        self.file_in_name = file_in
        self.ignore_list = ignore_list_in

        print("self.ignore_list " + str(self.ignore_list))
        self.data_frame_all = pandas.read_csv(file_in).fillna(value=0)

        self.data_frame = self.data_frame_all

        self.data_frame = self.data_frame_all.drop(self.ignore_list, 1)

        self.data_frame_ignore_frame = self.data_frame_all[self.ignore_list]
        #print("self.data_frame_ignore_frame " + str(self.data_frame_ignore_frame))

        #print("self.data_frame_ignore_frame t ")

        print(type(self.data_frame_ignore_frame))

        self.data_frame_ignore_frame_list = self.data_frame_ignore_frame.values.tolist()

        #print("self.data_frame_ignore_frame_list " + str(self.data_frame_ignore_frame_list))

        self.csv_column_name_list = list(self.data_frame.columns)
        self.encoded = False


    def write_ohe_csv(self, file_out_name):
        """
        opens file and writes one hot encoded data


Parameters:
----------
file_out_name: file
     Name of File to Write encoded categories to


Returns:
----------
    csv file output encoded using one-hot one-of-K encoding scheme


        """

        with open(file_out_name, "w") as myfile:
            writer = csv.writer(myfile)
            myarr = np.array(self.ignore_list)

            arr_flat = np.append(self.header, myarr)

            new_header = arr_flat.tolist()

            writer.writerow(new_header)
            i = 0

            for row in self.list_of_list:
                row_int = [int(i) for i in row]
                new_row = row_int + self.data_frame_ignore_frame_list[i]
                writer.writerow(new_row)
                i = i + 1

    def one_hot_encode(self):
        """
        opens file and writes one hot encoded data


Parameters:
----------
self: self
     self


Returns:
----------
    csv file output encoded using one-hot one-of-K encoding scheme


        """
        if self.encoded:
            return self.data_frame, self.csv_column_name_list

        self.enc = OneHotEncoder(handle_unknown='ignore')

        #print("one_hot_encode-- --- START ")

        self.enc.fit(self.data_frame)
        self.x_train_one_hot = self.enc.transform(self.data_frame)

        self.header = self.enc.get_feature_names(self.csv_column_name_list)

        self.ndarray = self.x_train_one_hot.toarray()

        self.list_of_list = self.ndarray.tolist()
        self.encoded = True
        return self.data_frame, self.csv_column_name_list


class OneHotEncoderBuilder():
    """
    opens file and writes one hot encoded data


Parameters:
----------
file_out_name: file
    Name of File to Write encoded categories to


Returns:
----------
csv file output encoded using one-hot one-of-K encoding scheme


    """

    def __init__(self, filename):
        """
        opens file and writes one hot encoded data


Parameters:
----------
    file_out_name: file
        Name of File to Write encoded categories to


Returns:
----------
    csv file output encoded using one-hot one-of-K encoding scheme


        """
        if filename is None:
            raise Exception("Filename cannot be none")
        self.filename = filename
        self.ignore_list = []

    def ignore(self, ignore):
        """
        constructs ignore list


Parameters:
----------
    ignore: string
        constructs ignore list


Returns:
----------
    constructs ignore list


        """
        self.ignore_list.append(ignore)
        return self

    def build(self):
        """
        builds OHE class

Parameters:
----------
        self: self
            self


Returns:
----------
        builds OHE class


        """
        return VLOneHotEncoder(self.filename, self.ignore_list)
