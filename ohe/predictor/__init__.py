import csv
import sys
import importlib
import pkgutil
from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder

class OneHotPredictor(ABC):

    @staticmethod
    def get_accuracy(y_pred_one_hot, y_test):
        correct = 0
        test_len = len(y_test)
        y_pred_one_hot_list = list(y_pred_one_hot)
        y_test_list = list(y_test)
        for i in range(test_len):
            if y_pred_one_hot_list[i] == y_test_list[i]:
                correct = correct + 1
        return ((correct / test_len) * 100)

    def __init__(self, target, X_test, X_train, y_test, y_train): #predict_list, training_test_split_percent):
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

    def __init__(self, target, training_test_split_percent, data_frame):
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
        if self.X_test is not None:
            raise Exception("Cannot add features after building predictor")
        if feature == self.target:
            raise Exception("Cannot have target as feature, cyclical prediction.")
        self.features.append(feature)
        return self

    def _split_data_frame(self, data_frame):
        # If we have already split the data_frame, we don't need to split it again
        if self.X_test is not None:
            return
        split_percent = self.training_test_split_percent / 100
        Y = data_frame[self.target]
        train_len = int(round(Y.size * split_percent))
        X = data_frame[self.features]
        # TODO: Shuffle before split
        X_train = X.iloc[:train_len]
        X_test = X.iloc[train_len:]

        enc = sk_OneHotEncoder(handle_unknown='ignore')
        enc.fit(X_train.values)

        self.X_train = enc.transform(X_train.values)
        self.X_test = enc.transform(X_test.values)
        self.y_train = Y.iloc[:train_len]
        self.y_test = Y.iloc[train_len:]

    def build(self, Constructor):
        self._split_data_frame(self.data_frame)
        return Constructor(self.target,
                           self.X_test,
                           self.X_train,
                           self.y_test,
                           self.y_train)


class Runner(object):

    def __init__(self, builder, algorithms):
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

        self.results.sort(key=lambda r: r['accuracy'], reverse=True)
        return self.results

    def write_predict_csv(self, file_out_name):

        if self.results is None:
            self.run_and_build_predictions()

        headers = [ r['model_name'] for r in self.results ]
        values = [ r['accuracy'] for r in self.results ]
        with open(file_out_name, mode='w') as _file:
            _writer = csv.writer(_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
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