from sklearn.tree import DecisionTreeClassifier
from ohe.predictor import OneHotPredictor, Commandline
from ohe.config import get_ohe_config
from ohe.predictor.algorithm.simple_predictor import SimplePredictor

model_name = 'Decision Tree Classifier'
algorithm = lambda: DecisionTreeClassifier(random_state=get_ohe_config().DTC_random_state)

@Commandline("DTC")
class DTC_OHP(SimplePredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        super().__init__(target, X_test, X_train, y_test, y_train, model_name, algorithm)