import csv
import sys
import importlib
import pkgutil
from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder

class OneHotPredictor(ABC):
    """
Overview
The optimal machine learning algorithm selector determines the highest accuracy technique in predicting a target given the same input features.
It is a Dynamic Dispatch Architecture using the Design Patterns: Builder, Factory, and Decorator Function Registration.

The machine learning algorithms that are currently optimized are:

Support Vector Machines
Logical Regression
AdaBoost Classifier
Gaussian Process Classifier
K Nearest Neighbors
Random Forest
Multi Layer Perceptron Neural Net
Quadratic Discriminant Analysis
Gaussian Naive Bayes
Decision Tree Classifier


To add a new ML algorithm to OHE

create a new python class file in the algorithm directory using this design pattern:

from ohe.predictor import OneHotPredictor, Commandline
from ohe.config import get_ohe_config

@Commandline("NEW_MACHINE_LEARNING_ALGORITHM_INITIALS")
class NEW_MACHINE_LEARNING_ALGORITHM_INITIALS_OHP(OneHotPredictor):

   def __init__(self, target, X_test, X_train, y_test, y_train):
       super().__init__(target, X_test, X_train, y_test, y_train)
       self.model_name = 'NEW MACHINE LEARNING ALGORITHM NAME'

   def predict(self):
       algorithm = NewMac()
       algorithm.fit(self.X_train.toarray(), self.y_train)
       y_pred = list(algorithm.predict(self.X_test.toarray()))
       self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
       return self.acc



The hyper parameters of the ML algorithms are driven from a config file that are iteratively optimized.

Add the NEW_MACHINE_LEARNING_ALGORITHM_INITIALS to the run command input predictor :
python -m ohe  \
 --file_in csvs/Married_Dog_Child_ID_Age.csv \
 --file_out_ohe csvs/Married_Dog_ID_Age_OHE.csv  \
 --file_out_predict csvs/Married_Dog_PREDICT_V2.csv \
 --file_in_config config/ohe_config.yaml \
 --ignore ID \
 --ignore Age \
 --target Kids \
 --training_test_split_percent 70 \
 --predictor SVM \
 --predictor LR \
 --predictor RF \
 --predictor MLP \
 --predictor GPC \
 --predictor QDA \
 --predictor KNN \
 --predictor GNB \
 --predictor DTC \
 --predictor ADA

 *********
 --preditor NEW_MACHINE_LEARNING_ALGORITHM_INITIALS


    """

    @staticmethod
    def get_accuracy(y_pred_one_hot, y_test):
        """
        opens file and writes one hot encoded data

        :param y_pred_one_hot: array - predicted values
        :param y_test: array - actual values
        :returns accuracy: float

        """
        correct = 0
        test_len = len(y_test)
        y_pred_one_hot_list = list(y_pred_one_hot)
        y_test_list = list(y_test)
        for i in range(test_len):
            if y_pred_one_hot_list[i] == y_test_list[i]:
                correct = correct + 1
        return ((correct / test_len) * 100)

    def __init__(self, target, X_test, X_train, y_test, y_train):
        """
        opens file and writes one hot encoded data

        :param target: string - label to be predicted or classified
        :param X_test: array(float) - testing features
        :param X_train: array(float) - training features
        :param y_test: array(float) - testing label
        :param y_train: array(float) - testing label
        """
        self.target = target
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train
        self.acc = -1
        self.model_name = "Undefined"

    @abstractmethod
    def predict(self):
        raise Exception("Not yet implemented.")
        pass

class OneHotPredictorBuilder(object):
    """
    reads the data frame and splits into training and test vectors
    using the splpit percentage from the command line
    the input to this OneHotPredictorBuilder is a csv fiule that gets parced to an array of integers or strings,
    denoting the values taken on by categorical features. The features are encoded using a one-hot ‘one-of-K’ encoding scheme.
    This creates a binary column for each category and returns a sparse matrix or dense array
    the encoder derives the categories based on the unique values in each feature.

     When features are categorical.
     For example a person could have features
     ["male", "female"],
     ["from Europe", "from US", "from Asia"],
     ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"].
     Such features can be efficiently coded as integers,
     for instance ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3]
     while ["female", "from Asia", "uses Chrome"] would be [1, 2, 1].

     """

    def __init__(self, target, training_test_split_percent, data_frame):
        """
        initializes
        :param target: string
        :param training_test_split_percent: string
        :param data_frame: panda frame


          """
        if target == None:
            raise Exception("target cannot be none")
        self.target = target
        self.training_test_split_percent = training_test_split_percent
        self.features = []
        self.data_frame = data_frame
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None

    def add_feature(self, feature):
        """
        add feature
        :param feature: string

          """
        if self.X_test is not None:
            raise Exception("Cannot add features after building predictor")
        if feature == self.target:
            raise Exception("Cannot have target as feature, cyclical prediction.")
        self.features.append(feature)
        return self

    def _split_data_frame(self, data_frame):
        """
          splits feature input data array of floats into training and test parts
          depending onn the split percent variable

          :param data_frame: panda frame
          :returns accuracy: float

          """
        from sklearn.utils import shuffle
        if self.X_test is not None:
            return
        split_percent = self.training_test_split_percent / 100


        Y_pre_shuffle = data_frame[self.target]
        train_len = int(round(Y_pre_shuffle.size * split_percent))
        X_pre_shuffle = data_frame[self.features]

        X, Y = shuffle(X_pre_shuffle, Y_pre_shuffle, random_state=13)

        X_train = X.iloc[:train_len]
        X_test = X.iloc[train_len:]

        enc = sk_OneHotEncoder(handle_unknown='ignore')
        enc.fit(X_train.values)

        self.X_train = enc.transform(X_train.values)
        self.X_test = enc.transform(X_test.values)
        self.y_train = Y.iloc[:train_len]
        self.y_test = Y.iloc[train_len:]

    def build(self, Constructor):
        """

        calls the constructor of the python machine learning algorithm class
        and passes the
        array of floats into training and test parts already split


        """
        self._split_data_frame(self.data_frame)
        return Constructor(self.target,
                           self.X_test,
                           self.X_train,
                           self.y_test,
                           self.y_train)


class Runner(object):
    """

    class pulls together builder object that has data frame and all algorithm from command line
    and algorithm directory
    then trains the ML functions
    then runs the ML predictors
    then checks accuracy then writes the ML and accuracy to file
    left most predictors is optimal

    """

    def __init__(self, builder, algorithms):
        """

        :param builder: class that reads data and encodes it for use in algorithms
        :param algorithms: library of scikit learn machine learning models
        """
        self.builder = builder
        self.algorithms = algorithms
        self.results = None


    def run_and_build_predictions(self):
        if self.results is not None:
            return self.results
        self.results = []

        for alg in self.algorithms:
            result = {}
            predictor = self.builder.build(alg)
            result['model_name'] = predictor.model_name
            result['accuracy'] = predictor.predict()
            self.results.append( result )

        self.results.sort(key=lambda r: r['model_name'], reverse=False)


        return self.results

    def write_predict_csv(self, file_out_name,target,write_header_flag=1):
        """
        write csv file with configured python machine learning algorithm and accuracy
        :param file_out_name: string
        """
        if self.results is None:
            self.run_and_build_predictions()

        headers = [ r['model_name'] for r in self.results ]
        headers.append("Target")

        values = [ r['accuracy'] for r in self.results ]
        values.append(target)
        import string



        with open(file_out_name, mode='a') as _file:
            _writer = csv.writer(_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if(write_header_flag == 1):
                _writer.writerow(headers)
            _writer.writerow(values)


### Below is a bit of "magic" to make the Commandline decorator work.

__algorithm_lookup =  {}

def Commandline(arg_name, **kwargs):
    """
    Decorator for OneHotPredictor classes.
    Given a subclass of OneHotPredictor, add it to the algorithm lookup table
    """
    if arg_name in __algorithm_lookup:
        raise Exception(f"Duplicate Commandline argument definition for {arg_name} found.")

    def wrap(klass):
        __algorithm_lookup[arg_name] = klass
        return klass
    return wrap

def get_algorithm_from_string(command_line_arg):
    if command_line_arg not in __algorithm_lookup:
        raise Exception(f"No algorithm found for {command_line_arg}.")
    return __algorithm_lookup[command_line_arg]


def __import_submodules(package_name):
    """ Import all submodules of a module, recursively
    :param package_name: Package name
    :type package_name: str
    :rtype: dict[types.ModuleType]
    """
    package = sys.modules[package_name]
    return {
        name: importlib.import_module(package_name + '.' + name)
        for loader, name, is_pkg in pkgutil.walk_packages(package.__path__)
    }

# Loads all of the algorithm classes from this packages
import ohe.predictor.algorithm
__import_submodules('ohe.predictor.algorithm')