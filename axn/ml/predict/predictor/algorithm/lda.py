"""
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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from axn.ml.predict.predictor import Commandline
from axn.ml.predict.predictor.algorithm.simple_predictor import SimplePredictor

model_name = 'Linear Discriminant Analysis'
def algorithm(): return LinearDiscriminantAnalysis()


@Commandline("LDA")
class LDA_OHP(SimplePredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        """
         initializes the training and testing features and labels

         :param target: string - label to be predicted or classified
         :param X_test: array(float) - testing features
         :param X_train: array(float) - training features
         :param y_test: array(float) - testing label
         :param y_train: array(float) - testing label
         """
        super().__init__(target, X_test, X_train, y_test, y_train, model_name, algorithm)
