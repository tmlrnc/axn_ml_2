from sklearn.tree import DecisionTreeClassifier
from axn.ml.predict.predictor import OneHotPredictor, Commandline
from axn.ml.predict.config import get_ohe_config
from axn.ml.predict.predictor.algorithm.simple_predictor import SimplePredictor

model_name = 'Decision Tree Classifier'


def algorithm(): return DecisionTreeClassifier(
    random_state=get_ohe_config().dtc_random_state)


@Commandline("DTC")
class DTC_OHP(SimplePredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        super().__init__(target, X_test, X_train, y_test, y_train, model_name, algorithm)
