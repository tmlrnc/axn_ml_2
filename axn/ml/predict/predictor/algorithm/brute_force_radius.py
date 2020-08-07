"""
Bayesian RIDGE REGRESSION

<img src="images/bay.png" alt="OHE">

In general, when fitting a curve with a polynomial by Bayesian ridge regression, the selection of initial values of the
regularization parameters (alpha, lambda) may be important. This is because the regularization parameters are determined
by an iterative procedure that depends on initial values.
In this example, the sinusoid is approximated by a polynomial using different pairs of initial values.
When starting from the default values (alpha_init = 1.90, lambda_init = 1.), the bias of the resulting curve is large,
and the variance is small. So, lambda_init should be relatively small (1.e-3) so as to reduce the bias.
Also, by evaluating log marginal likelihood (L) of these models, we can determine which one is better.
It can be concluded that the model with larger L is more likely.

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
from sklearn.neighbors import RadiusNeighborsRegressor

from axn.ml.predict.predictor import OneHotPredictor, Commandline
from axn.ml.predict.config import get_ohe_config


@Commandline("BFRA")
class BruteForceRadius_OHP(OneHotPredictor):

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
        self.model_name = 'BFRA'

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
        algorithm = RadiusNeighborsRegressor(
            radius=get_ohe_config().rnr_radius)
        algorithm.fit(self.X_train, self.y_train)
        y_pred = list(algorithm.predict(self.X_test))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc
