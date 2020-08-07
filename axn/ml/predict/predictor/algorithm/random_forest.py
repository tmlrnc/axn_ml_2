"""
Random forest

<img src="images/rf.png" alt="OHE">

like its name implies, consists of a large number of individual decision trees
that operate as an ensemble. Each individual tree in the random forest spits out a
class prediction and the class with the most votes becomes our model’s prediction
A large number of relatively uncorrelated models (trees) operating as a
committee will outperform any of the individual constituent models.

The low correlation between models is the key. Just like how investments with low correlations
(like stocks and bonds) come together to form a portfolio that is greater than the sum of its parts,
uncorrelated models can produce ensemble predictions that are more accurate than any of the
individual predictions. The reason for this wonderful effect is that the trees protect each other
from their individual errors (as long as they don’t constantly all err in the same direction).
While some trees may be wrong, many other trees will be right, so as a group the trees are able to
move in the correct direction. So the prerequisites for random forest to perform well are:

There needs to be some actual signal in our features so that models built using those features do better than random guessing.
The predictions (and therefore the errors) made by the individual trees need to have low correlations with each other.


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
from sklearn.ensemble import RandomForestClassifier

from axn.ml.predict.predictor import OneHotPredictor, Commandline
from axn.ml.predict.config import get_ohe_config


@Commandline("RF")
class RandomForest_OHP(OneHotPredictor):

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
        self.model_name = 'Random Forest'

    def predict(self):
        """
Random forest

like its name implies, consists of a large number of individual decision trees
that operate as an ensemble. Each individual tree in the random forest spits out a
class prediction and the class with the most votes becomes our model’s prediction
A large number of relatively uncorrelated models (trees) operating as a
committee will outperform any of the individual constituent models.

The low correlation between models is the key. Just like how investments with low correlations
(like stocks and bonds) come together to form a portfolio that is greater than the sum of its parts,
uncorrelated models can produce ensemble predictions that are more accurate than any of the
individual predictions. The reason for this wonderful effect is that the trees protect each other
from their individual errors (as long as they don’t constantly all err in the same direction).
While some trees may be wrong, many other trees will be right, so as a group the trees are able to
move in the correct direction. So the prerequisites for random forest to perform well are:

There needs to be some actual signal in our features so that models built using those features do better than random guessing.
The predictions (and therefore the errors) made by the individual trees need to have low correlations with each other.


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
        algorithm = RandomForestClassifier(
            n_estimators=get_ohe_config().rf_estimators,
            max_depth=get_ohe_config().rf_max_depth)
        algorithm.fit(self.X_train, self.y_train)
        y_pred = list(algorithm.predict(self.X_test))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc
