"""


SVM

<img src="images/svm.png" alt="OHE">



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

support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:


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

from sklearn.svm import LinearSVC

from axn.ml.predict.predictor import OneHotPredictor, Commandline
from axn.ml.predict.config import get_ohe_config


@Commandline("SVM")
class SVM_OHP(OneHotPredictor):

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
        self.model_name = 'Linear SVC'

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
        algorithm = LinearSVC(
            random_state=get_ohe_config().lr_random_state,
            max_iter=-1,
            multi_class=get_ohe_config().s_multi_class)
        algorithm.fit(self.X_train, self.y_train)
        y_pred = list(algorithm.predict(self.X_test))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc
