import csv
from sklearn.preprocessing import OneHotEncoder
import pandas
import numpy as np

# TODO: Add documentation
class OneHotEncoder(object):

    def __init__(self, file_in,ignore_list_in):
        self.file_in_name = file_in
        self.ignore_list = ignore_list_in
        self.data_frame_all = pandas.read_csv(file_in)
        for ignore in self.ignore_list:
            self.ignore = ignore
            self.data_frame = self.data_frame_all.drop(ignore, 1)
        self.data_frame_all_ignore = self.data_frame_all[self.ignore]
        self.data_frame_all_ignore_list = self.data_frame_all_ignore.tolist()
        self.csv_column_name_list = list(self.data_frame.columns)
        self.encoded = False

        return

    def write_ohe_csv(self,file_out_name):
        self.one_hot_encode()
        with open(file_out_name, "w") as f:
            writer = csv.writer(f)
            myarr = np.array(self.ignore)
            arr_flat = np.append(self.header,myarr)
            writer.writerow(arr_flat)
            i = 0
            for row in self.listOflist:
                ignore_value = self.data_frame_all_ignore_list[i]
                row.append(ignore_value)
                writer.writerow(row)
                i = i + 1


    def one_hot_encode(self):
        if self.encoded:
            return self.data_frame, self.csv_column_name_list

        from sklearn.preprocessing import OneHotEncoder
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(self.data_frame)
        self.X_train_one_hot = self.enc.transform(self.data_frame)

        self.header = self.enc.get_feature_names(self.csv_column_name_list)
        self.ndarray = self.X_train_one_hot.toarray()
        self.listOflist = self.ndarray.tolist()
        self.encoded = True
        return self.data_frame, self.csv_column_name_list




class OneHotEncoderBuilder(object):
    def __init__(self, filename):
        if filename == None:
            raise Exception("Filename cannot be none")
        self.filename = filename
        self.ignore_list = []

    def ignore(self, ignore):
        self.ignore_list.append(ignore)
        return self

    def build(self):
        return OneHotEncoder(self.filename, self.ignore_list)
