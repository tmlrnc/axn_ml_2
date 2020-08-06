"""
AUTOMATED MACHINE LEARNING  PROCESS

<img src="images/class.png" alt="DIS">


Step 1 Data Read
----------
Data Read from CSV or URL or API to local CSV
CSV columns are features and target column
Read File
Read API
The quantity & quality of your data dictate how accurate our model is. Datasets: for test datasets and for generating datasets with
specific properties for investigating model behavior
Read URL to CSV



Step 2 Data Preparation
----------
Data Preparation
Maximum Density -is there enough data?
Maximum Variance - standard normally distributed Gaussian data: with zero mean and unit variance.
Maximum Diversity - is it not all the same?
Minimum Noise - is it not all random ?
Normalize - If a feature has a variance that is orders of magnitude larger than others, it might dominate the objective function and make the
estimator unable to learn from other features correctly as expected.
Scale - Remove the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.
Complexity - scaling features to lie between a given minimum and maximum value, often between zero and one, or so that the maximum absolute value of each feature is scaled to unit size
Anomaly Detection and Elimination Novelty detection - compute mean and standard deviation and verify Non exponential
Feature Extraction For defining attributes of data. - find true meaning of data
Feature Definition: identify predictive attributes from which to create supervised models.
Dimensionality Reduction: for reducing the number of attributes in data for summarization, visualization and feature selection such as Principal component analysis.
Genie Data Feature Selection and Dimensionality Reduction We start with preparing the data. Given a set of features F = { cases, population, density, age,
current health conditions, medical resources available,  ,…, fi ,…, fn } the GNY Feature Selection’s goal is to find a
subset of features that maximizes GNY models ability to classify patterns.The product of GNY’s Feature Selection will maximize our predictability scoring functions
(F1, accuracy, recall, precision) for subsequent algorithm analysis and prediction. The GNY Feature Selection algorithms also perform data mapping and transformation. However, this can only be performed with subsets of the entire group of input variables. This process is referred to Dimensionality Reduction. Dimensionality Reduction is essential because it is computationally intractable to search the whole selection of possible feature subsets so GNY uses approximations of the optimal subset. GNY uses efficient search-heuristics and machine learning to create these approximations and optimal subsets.
Principal Component Analysis - The eigenvectors and eigenvalues of a covariance matrix are the principal components – the eigenvectors determine the directions of the new feature space, and the eigenvalues determine their magnitude. The goal PCA is to reduce the dimensionality of the original feature space by projecting it onto a smaller subspace, where the eigenvectors will form the axes, So how many principal components are we going to choose for our new feature subspace? Genie users the explained variance, calculated from the eigenvalue, which tells us how much information (variance) can be attributed to each of the principal components. So for the two variables shown below x1 = age and x2 = gender we are able to reduce them to a component axes that maximizes the variance – data for each variable. With LDA for example we would maximize the component for data separation.
Neural Net Automated Feature selection - performed by genie’s recurrent neural networks in conjunction with its L1-regularization.
Genie implements this using the weight decay technique with the corresponding protocol. This genie feature selection technique is a part of our neural network pruning and contrasting process. The network self-selects features from the metadata by using its backpropagation learning algorithm to weight and bias the features most needed.

Data Anonymizer Genie removes personal identity data and assigns a universal Genie ID that keeps behavior behind the numbers unchanged.
Genie also randomly samples numerical variables from probability distributions of personal data. Genie further anonymizes personal numerical data by finding one feature that is the most important, uses our best-fit distribution, then randomly samples these values. Once Genie has this variable simulated, we use Cholesky decomposition to add multiple correlated columns to our data. Genie’s proprietary Cholesky decomposition is derived from financial Monte Carlo methods to simulate systems with multiple correlated variables.
Data Historical Classifiers
Read Historical Neural Net Data Classifier Layer Node Weights and Biases
Genie uses a high number of hidden nodes; the number of unknown weights approaches the number of training equations; therefore Genie solves this problem by introducing and saving a new hidden layer which reduces the total number of nodes. Genie reads historical classifiers back and adjusts the weights and biases, so that we optimize the cost function. Genie adjusts the whole neural network, so that the output value is optimized. This is how we tell the net that it performed poorly or well. We keep trying to optimize the cost function by running through new observations from our dataset. To update the network, we calculate gradients, which are small updates to individual weights in each saved layer.
Data Encode Classifier  Data Discretize Continuous
Every data feature column you want to ignore you must mark with ignore flag
For each continuous data feature column that you do not ignore. YOU MUST discretize it using one of 4 methods
One Hot Encode the transformed result with one-hot encoding and return a sparse matrix. Ignored features are always stacked to the right.
One Hot-dense - Encode the transformed result with one-hot encoding and return a dense array. Ignored features are always stacked to the right
Ordinal Return the bin identifier encoded as an integer value.
Strategy used to define the widths of the bins.
Uniform All bins in each feature have identical widths.
Quantile All bins in each feature have the same number of points.
K Means Values in each bin have the same nearest center of a 1D k-means cluster.
For every category One Hot Encodes the features you do not ignore.
Feature Selection for Predictive Power
Target Section
Model Integration
Translate Models from Python to NodeJS
Integrated Models into Genie Distributed
10 are done



Step 3 Model Selection
----------
Model Selection
Clustering: for grouping unlabeled data such as KMeans
Regression - Predicting a continuous-valued attribute associated with an object.
Classifiers Identifying which category an object belongs to. Classification is the step in the record linkage
process where record pairs are classified into matches, non-matches and possible matches. Classification algorithms can be supervised or unsupervised (with or without training data).
Cross Validation: for estimating the performance of supervised models on unseen data.
Ensemble methods: for combining the predictions of multiple supervised models.
Manifold Learning: For summarizing and depicting complex multi-dimensional data.
Supervised Models: a vast array not limited to generalized linear models, discriminant analysis, naive Bayes,
lazy methods, neural networks, support vector machines and decision trees.


Step 4 Train the Models
----------
Train the Models
Split Feature Data into Feature Training and Feature Test Sets
Train Models with Feature Training Data Set
Run Model Predictions with Feature Test Data Set
Measure Models - Predicted Target Value vs Actual Target Value
Accuracy
Precision
Recall
Confusion Matrix
True Positive - Pred Pos and Act Pos
True Negative
False Positive
False Negative
Optimize Models in a Feedback Iterative Loop
Optimize Model Hyperparameter
Optimize Feature Selection
Save Historical Classifiers



Step 5 Test Results
----------
OUTPUT: BEST MODEL WITH BEST DATA that BEST predictive Results in csv

The quantity & quality of your data dictate how accurate our model is. Datasets: for test datasets and for
generating datasets with specific properties for investigating model behavior.



"""
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
# pylint: disable=unreachable
# pylint: disable=too-many-instance-attributes
# pylint: disable=import-error


import csv
import sys
import importlib
import pkgutil
from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import axn.ml.predict.predictor.algorithm

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
    def get_recall_score(y_pred_one_hot, y_test):
        """
        opens file and writes one hot encoded data

        :param y_pred_one_hot: array - predicted values
        :param y_test: array - actual values
        :returns
      The recall is the ratio tp / (tp + fn) where tp is the number of true positives and
      fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
      The best value is 1 and the worst value is 0.


        """
        recall = 0
        try:
            recall = recall_score(
                y_test,
                y_pred_one_hot,
                average='micro',
                zero_division=1)
        except ValueError:
            print('Caught ValueError')
            return 0
        return recall

    @staticmethod
    def get_precision_score(y_pred_one_hot, y_test):
        """
        opens file and writes one hot encoded data

        :param y_pred_one_hot: array - predicted values
        :param y_test: array - actual values
        :returns
        The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
        The best value is 1 and the worst value is 0.

        """
        precision = 0
        try:
            precision = precision_score(y_test, y_pred_one_hot, average=None)
        except ValueError:
            print('Caught ValueError')
            return 0

        return precision

    @staticmethod
    def get_f1_score(y_pred_one_hot, y_test):
        """
        opens file and writes one hot encoded data

        :param y_pred_one_hot: array - predicted values
        :param y_test: array - actual values
        :returns The F1 score: float
        The F1 score can be interpreted as a weighted average of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:
        F1 = 2 * (precision * recall) / (precision + recall)

        """
        f1 = 0
        try:
            f1 = f1_score(y_test, y_pred_one_hot, average='weighted')
        except ValueError:
            print('Caught ValueError')
            return 0

        return f1

    @staticmethod
    def get_classification_accuracy_MARGIN(y_pred_one_hot, y_test):
        """
        opens file and writes one hot encoded data

        :param y_pred_one_hot: array - predicted values
        :param y_test: array - actual values
        :returns accuracy: float

        """
        correct = 0
        classification_accuracy = 0
        test_len = len(y_test)
        y_pred_one_hot_list = list(y_pred_one_hot)
        y_test_list = list(y_test)
        for i in range(test_len):

            _imargin = y_test_list[i] / 10
            # print(_imargin)
            _imargin_i = int(round(_imargin))

            _imargin_i_a = abs(y_pred_one_hot_list[i] - y_test_list[i])

            if _imargin_i_a <= _imargin_i:
                correct = correct + 1

        classification_accuracy = ((correct / test_len) * 100)
        return classification_accuracy

    @staticmethod
    def get_classification_accuracy(y_pred_one_hot, y_test):
        """
        opens file and writes one hot encoded data

        :param y_pred_one_hot: array - predicted values
        :param y_test: array - actual values
        :returns accuracy: float

        """
        correct = 0
        classification_accuracy = 0
        test_len = len(y_test)
        y_pred_one_hot_list = list(y_pred_one_hot)
        y_test_list = list(y_test)
        for i in range(test_len):
            if y_pred_one_hot_list[i] == y_test_list[i]:
                correct = correct + 1
        classification_accuracy = ((correct / test_len) * 100)
        return classification_accuracy

    @staticmethod
    def get_accuracy(y_pred_one_hot, y_test):
        """
        opens file and writes one hot encoded data

        :param y_pred_one_hot: array - predicted values
        :param y_test: array - actual values
        :returns accuracy: float





        """
        acc_dict = {}

        acc_dict["y_pred_one_hot"] = y_pred_one_hot
        acc_dict["y_test"] = y_test
        acc_dict["f1_score"] = OneHotPredictor.get_f1_score(
            y_pred_one_hot, y_test)
        acc_dict["classification_accuracy"] = OneHotPredictor.get_classification_accuracy_MARGIN(
            y_pred_one_hot, y_test)

        return acc_dict

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
        """
        will be used for special cases
        """
        raise Exception("Not yet implemented.")


class OneHotPredictorBuilder():
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

    def __init__(
            self,
            target,
            training_test_split_percent,
            data_frame,
            strategy):
        """
        initializes
        :param target: string
        :param training_test_split_percent: string
        :param data_frame: panda frame


          """
        if target is None:
            raise Exception("target cannot be none")
        self.target = target
        self.strategy = strategy
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
            raise Exception(
                "Cannot have target as feature, cyclical prediction.")
        self.features.append(feature)
        return self

    def _split_data_frame(self, data_frame):
        """
          splits feature input data array of floats into training and test parts
          depending onn the split percent variable

          :param data_frame: panda frame
          :returns accuracy: float

          """
        if self.X_test is not None:
            return
        split_percent = self.training_test_split_percent / 100

        Y_pre_shuffle = data_frame[self.target]
        train_len = int(round(Y_pre_shuffle.size * split_percent))

        X_pre_shuffle = data_frame[self.features]

        #X, Y = shuffle(X_pre_shuffle, Y_pre_shuffle, random_state=13)

        X = X_pre_shuffle
        Y = Y_pre_shuffle

        X_train = X.iloc[:train_len]
        X_test = X.iloc[train_len:]

        if self.strategy == "ohe":
            enc = sk_OneHotEncoder(handle_unknown='ignore')
            enc.fit(X_train.values)
            self.X_train = enc.transform(X_train.values)
            self.X_test = enc.transform(X_test.values)
        else:
            self.X_train = X_train
            self.X_test = X_test

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


class Runner():
    """


    <img src="images/ml.png" alt="DIS">

    class pulls together builder object that has data frame and all algorithm from command line
    and algorithm directory
    then trains the ML functions
    then runs the ML predictors
    then checks accuracy then writes the ML and accuracy to file
    left most predictors is optimal

    """

    # pylint: disable=redefined-builtin
    # pylint: disable=unused-argument
    # pylint: disable=redefined-outer-name
    # pylint: disable=import-outside-toplevel

    def __init__(self, builder, algorithms):
        """

        :param builder: class that reads data and encodes it for use in algorithms
        :param algorithms: library of scikit learn machine learning models
        """
        self.builder = builder
        self.algorithms = algorithms
        self.results = None
        self.predictions_list_of_dicts_of_predictions = []
        self.predictions_list_of_dicts_of_actuals = []
        self.my_model_headers = []
        self.model_list = []

    def run_and_build_predictions(
            self,
            score_list,
            my_file_out_predict,
            data_frame_ignore_frame,
            training_test_split_percent,
            ignore_list,
            write_actual_flag=1):
        """

        :param builder: class that reads data and encodes it for use in algorithms
        :param algorithms: library of scikit learn machine learning models
        """
        if self.results is not None:
            return self.results
        self.results = []
        # alg_results is a list of dictionaries, such that alg_results[0] is a dictionary for index 0
        # It contains a dictionary with { id, actual, pred_alg_1, pred_alg_2,
        # etc }
        alg_results = None
        headers = set(['id', 'actual'])

        print("full_len")

        print(ignore_list)
        for ig in ignore_list:
            myignore = ig
        split_percent = training_test_split_percent / 100
        full_len = data_frame_ignore_frame.size

        print("full_len")
        print(full_len)
        train_len = int(round(full_len * split_percent))

        print("train_len")
        print(train_len)
        data_frame_ignore_frame_X = data_frame_ignore_frame.iloc[train_len:]

        print("data_frame_ignore_frame_X")
        print(data_frame_ignore_frame_X)
        print("data_frame_ignore_frame_X size")
        print(data_frame_ignore_frame_X.size)

        data_frame_ignore_frame_X_d = data_frame_ignore_frame_X[myignore]

        data_frame_ignore_frame_X_d_list = data_frame_ignore_frame_X_d.to_list()

        for alg in self.algorithms:

            predictor = self.builder.build(alg)

            model_name = str(predictor.model_name)
            headers.add(model_name)
            acc_dict = predictor.predict()
            headers.add("UID")

            self.model_list.append(model_name)
            y_pred_one_hot = acc_dict['y_pred_one_hot']
            print("type(y_pred_one_hot)")

            print(type(y_pred_one_hot))
            y_test = acc_dict['y_test']
            y_test_list = y_test.tolist()

            # Initialize alg_results *IF* it has not been initialized yet
            if alg_results is None:
                alg_results = []
                for (act, ix) in zip(y_test_list, range(len(y_test_list))):
                    alg_results.append({'id': ix, 'actual': act})

                data_frame_ignore_frame_X_d_list_dict = dict(zip(
                    data_frame_ignore_frame_X_d_list, range(len(data_frame_ignore_frame_X_d_list))))
                for (pred, id) in data_frame_ignore_frame_X_d_list_dict.items():
                    print(pred)
                    print(id)
                    alg_results[id][myignore] = pred

            y_pred_one_hot_dict = dict(
                zip(y_pred_one_hot, range(len(y_pred_one_hot))))
            for (pred, id) in y_pred_one_hot_dict.items():
                # print(pred)
                # print(id)
                alg_results[id][model_name] = pred

            for key in acc_dict.keys():
                if key in score_list:
                    result = {}
                    result['model_name_score_name'] = model_name + "_" + key
                    result['score_value'] = acc_dict[key]
                    self.results.append(result)

        self.results.sort(
            key=lambda r: r['model_name_score_name'],
            reverse=False)

        with open(my_file_out_predict, 'w') as io:
            writer = csv.DictWriter(
                io,
                fieldnames=list(headers),
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(alg_results)

        return self.results

    def write_predict_csv(
            self,
            file_out_name_score,
            file_out_tableau,
            file_in_master,
            my_file_out_predict,
            target,
            write_header_flag=1):
        """
        write csv file with configured python machine learning algorithm and accuracy
        :param file_out_name: string
        """

    # read my_file_out_predict
    # read file_in_master
    # join UID


        import pandas as pd
        a_file_in_master = pd.read_csv(file_in_master)
        print("a_file_in_master")
        print(type(a_file_in_master))
        print(self.model_list)
        for mymode in self.model_list:
            mode_to_use = mymode

        b_my_file_out_predict = pd.read_csv(my_file_out_predict)

        merged = a_file_in_master.merge(b_my_file_out_predict, on='UID')

        merged_selected = merged[['Admin2',
                                  'Country_Region',
                                  'Province_State',
                                  mode_to_use,
                                  'actual']]
        merged_selected['Diff'] = merged_selected[mode_to_use] - \
            merged_selected['actual']

        merged_selected['Diff_Percent'] = 1 - \
            (merged_selected[mode_to_use] / merged_selected['actual'])
        merged_selected_renamed = merged_selected.rename(
            columns={'MLP': 'Predicted', 'actual': 'Actual'})
        print("Diff_Percent")

        for ind in merged_selected_renamed.index:
            if merged_selected_renamed['Actual'][ind] == 0:
                merged_selected_renamed['Diff_Percent'][ind] = 0

        # here is the simplist way to add the new column
        merged_selected_renamed['Date'] = target

        merged_selected_renamed.to_csv(file_out_tableau, mode='a', index=False)

        headers = [r['model_name_score_name'] for r in self.results]
        headers.append("Target")

        values_score = [r['score_value'] for r in self.results]
        values_score.append(target)
        with open(file_out_name_score, mode='a') as _file:
            _writer = csv.writer(
                _file,
                delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            if write_header_flag == 1:
                _writer.writerow(headers)
            _writer.writerow(values_score)


# Below is a bit of "magic" to make the Commandline decorator work.
__algorithm_lookup = {}


def Commandline(arg_name, **kwargs):
    """
    Decorator for OneHotPredictor classes.
    Given a subclass of OneHotPredictor, add it to the algorithm lookup table
    """
    # pylint: disable=unused-argument

    if arg_name in __algorithm_lookup:
        raise Exception(
            f"Duplicate Commandline argument definition for {arg_name} found.")

    def wrap(klass):
        __algorithm_lookup[arg_name] = klass
        return klass
    return wrap


def get_algorithm_from_string(command_line_arg):
    """

     :param builder: class that reads data and encodes it for use in algorithms
     :param algorithms: library of scikit learn machine learning models
     """
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
__import_submodules('axn.ml.predict.predictor.algorithm')
