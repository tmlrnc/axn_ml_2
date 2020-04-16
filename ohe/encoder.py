import csv
from sklearn.preprocessing import OneHotEncoder
import pandas
import numpy as np

class OneHotEncoder(object):
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
    def __init__(self, file_in,ignore_list_in):
        """
        opens file and writes one hot encoded data

        :param ignore_list_in: list[] : list of feature to ignore
        :param file_in: string : input data file
        """
        self.file_in_name = file_in
        self.ignore_list = ignore_list_in

        print("self.ignore_list " + str(self.ignore_list))
        self.data_frame_all = pandas.read_csv(file_in).fillna(value = 0)

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

        return

    def write_ohe_csv(self,file_out_name):
        """
        opens file and writes one hot encoded data

        :param file_out_name: Name of File to Write to
        """

        with open(file_out_name, "w") as f:
            writer = csv.writer(f)
            myarr = np.array(self.ignore_list)

            arr_flat = np.append(self.header,myarr)

            new_header = arr_flat.tolist()

            writer.writerow(new_header)
            i = 0
            print(type(self.listOflist))

            for row in self.listOflist:
                row_int = [int(i) for i in row]
                new_row = row_int + self.data_frame_ignore_frame_list[i]
                writer.writerow(new_row)
                i = i + 1


    def one_hot_encode(self):
        """
         runs OneHotEncoder() function on class data

        :returns data_frame: array
        :returns csv_column_name_list: array

         """
        if self.encoded:
            return self.data_frame, self.csv_column_name_list

        from sklearn.preprocessing import OneHotEncoder
        self.enc = OneHotEncoder(handle_unknown='ignore')

        #print("one_hot_encode-- --- START ")

        self.enc.fit(self.data_frame)
        self.X_train_one_hot = self.enc.transform(self.data_frame)



        self.header = self.enc.get_feature_names(self.csv_column_name_list)

        self.ndarray = self.X_train_one_hot.toarray()



        self.listOflist = self.ndarray.tolist()
        self.encoded = True
        return self.data_frame, self.csv_column_name_list




class OneHotEncoderBuilder(object):
    def __init__(self, filename):
        """
        opens file and writes one hot encoded data

        :param filename: string : input data file
        """
        if filename == None:
            raise Exception("Filename cannot be none")
        self.filename = filename
        self.ignore_list = []

    def ignore(self, ignore):
        """
        constructs ignore list

        :param ignore: string : one feature string on list
        """
        self.ignore_list.append(ignore)
        return self

    def build(self):
        """
        builds OHE class

        :returns OneHotEncoder: class

        """
        return OneHotEncoder(self.filename, self.ignore_list)
