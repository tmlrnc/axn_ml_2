"""
 KN RREGRESSOR

<img src="images/knn.png" alt="OHE">

provides functionality for unsupervised and supervised neighbors-based learning methods.
Unsupervised nearest neighbors is the foundation of many other learning methods, notably
manifold learning and spectral clustering. Supervised neighbors-based learning comes in two flavors:
classification for data with discrete labels, and regression for data with continuous labels.

The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point,
and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the
local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard
Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods,
since they simply “remember” all of its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree).

Despite its simplicity, nearest neighbors has been successful in a large number of classification and regression problems,
including handwritten digits and satellite image scenes. Being a non-parametric method, it is often successful in classification
situations where the decision boundary is very irregular.

The classes in sklearn.neighbors can handle either NumPy arrays or scipy.sparse matrices as input. For dense matrices, a
large number of possible distance metrics are supported. For sparse matrices, arbitrary Minkowski metrics are supported for searches.

There are many learning routines which rely on nearest neighbors at their core. One example is kernel density estimation,
discussed in the density estimation section.

Parameters
----------
X_test: array
    testing features

X_train: array
    training features

y_test: array
    testing label

y_train: array
    testing label

Returns:
----------
    target: array - label to be predicted or classified

 trains the scikit-learn  python machine learning algorithm library function
 https://scikit-learn.org

 then passes the trained algorithm the features set and returns the
 predicted y test values form, the function

 then compares the y_test values from scikit-learn predicted to
 y_test values passed in

 then returns the accuracy
 """
# pylint: disable=duplicate-code
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name





from sklearn.neighbors import KNeighborsRegressor
from axn.ml.predict.predictor import OneHotPredictor, Commandline
from axn.ml.predict.config import get_ohe_config


@Commandline("KNRREGRESSOR")
class KNeighborsRegressor_OHP(OneHotPredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        """
        initializes the training and testing features and labels

        :param target: string - label to be predicted or classified
        :param X_test: array(float) - testing features
        :param X_train: array(float) - training features
        :param y_test: array(float) - testing label
        :param y_train: array(float) - testing label
        """
        super().__init__(target, X_test, X_train, y_test, y_train)
        self.model_name = 'KNRREGRESSOR'

    def predict(self):
        """
        trains the scikit-learn  python machine learning algorithm library function
        https://scikit-learn.org

        then passes the trained algorithm the features set and returns the
        predicted y test values form, the function

        then compares the y_test values from scikit-learn predicted to
        y_test values passed in

        then returns the accuracy
        """
        algorithm = KNeighborsRegressor(n_neighbors=3)
        algorithm.fit(self.X_train, self.y_train)
        y_pred = list(algorithm.predict(self.X_test))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc
