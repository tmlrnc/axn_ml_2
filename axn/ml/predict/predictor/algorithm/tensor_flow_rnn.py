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

from sklearn.cluster import KMeans
from axn.ml.predict.predictor import OneHotPredictor, Commandline
from axn.ml.predict.config import get_ohe_config
import numpy
import logging
import numpy
import random
from collections import defaultdict

from pykalman import KalmanFilter
import numpy as np


class VL_Tensor_Flow_RNN(object):
    """
The KF time series process begins by specifying a state space model, which relates your observations (the covid death time series per city in the PAST)
to the unobserved states (the covid death time series per city TOMORROW) and describes how the state evolves over time; this will give you your transition
and observation matrix as well as the covariance matrix of the state error this is called process noise, and the covariance matrix for the observation error.
Our Kalman FIlter is an algorithm for estimating the unobservable state (the future covid deaths per city)
and its variance-covariance matrix at each time once you've specified all those things.

Kalman filtering, also known as linear quadratic estimation, LQE, is an algorithm that uses a series of measurements observed over time,
containing statistical noise and other inaccuracies, and produces estimates of unknown variables that tend to be more accurate than those based
on a single measurement alone, by estimating a joint probability distribution over the variables for each timeframe.

The Kalman filter represents all distributions by Gaussians and iterates over two different things: measurement updates and motion updates.
Measurement updates involve updating a prior with a product of a certain belief, while motion updates involve performing a convolution.
Measurement updates use Bayes Rule. Imagine we’ve localized another covid city death rate, and have a prior distribution with a very high variance
(large uncertainty). If we get another measurement that tells us something about that covid city death rate with a smaller variance.
If we create a new gaussian by combining the information, the mean will be somewhere in between the two distributions, with a higher peak and narrower variance than the prior.


    """

    def __init__(self):
        return

    def updated_mean(self, mean1, var1, mean2, var2):
        """
  Update the prior mean with the weighted sum of the old means,
  where the weights are the variances of the other mean. The mean is normalized by the sum of the weighting factors.
        """

        new_mean = (mean1 * var2 + mean2 * var1) / (var1 + var2)
        return new_mean

    def updated_var(self, var1, var2):
        """
Update of the variance uses the previous variances.
        """
        new_var = 1 / ((1 / var1) + (1 / var2))
        return new_var

    def predict(self, X_test_data):
        Y_predicted_data = X_test_data.to_numpy()
        return Y_predicted_data

    def fit_KF(self, X_train_data, Y_train_data):

        print("X_train_data " + str(type(X_train_data)))
        # get mean and variance of X_train_data (mean1,var1) and mean and
        # variance of Y_train_data (mean2,var2)
        mean1 = X_train_data.mean()
        var1 = X_train_data.var()

        mean2 = Y_train_data.mean()
        var2 = Y_train_data.var()
        #self.MY_train_data = Y_train_data

        kf = KalmanFilter(transition_matrices=[[1, 1], [0, 1]], observation_matrices=[
                          [0.1, 0.5], [-0.3, 0.0]])
        measurements = np.asarray([[1, 0], [0, 0], [0, 1]])  # 3 observations
        kf = kf.em(measurements, n_iter=5)
        (filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
        (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

        return self

    def fit(self, X_train_data, Y_train_data):

        print("X_train_data " + str(type(X_train_data)))
        # get mean and variance of X_train_data (mean1,var1) and mean and
        # variance of Y_train_data (mean2,var2)
        mean1 = X_train_data.mean()
        var1 = X_train_data.var()

        mean2 = Y_train_data.mean()
        var2 = Y_train_data.var()
        #self.MY_train_data = Y_train_data

        return self

    def predict_KF(self, mean1, var1, mean2, var2):
        new_mean = mean1 + mean2
        new_var = var1 + var2
        return [new_mean, new_var]


@Commandline("TENSORFLOWRNN")
class Kalman_Filter(OneHotPredictor):

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
        self.model_name = 'TENSORFLOWRNN'

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
        algorithm = VL_Tensor_Flow_RNN()
        algorithm.fit(self.X_train, self.y_train)

        y_pred = list(algorithm.predict(self.X_test))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc
