from sklearn.neural_network import MLPRegressor

from predict.predictor import OneHotPredictor, Commandline
from predict.config import get_ohe_config

@Commandline("OMPCVCONT")
class MLPRegression_OHP(OneHotPredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        super().__init__(target, X_test, X_train, y_test, y_train)
        self.model_name = 'OMPCVCONT'

    def predict(self):
        algorithm = MLPRegressor(hidden_layer_sizes=(20, 20), learning_rate_init=0.01,early_stopping=True)
        algorithm.fit(self.X_train, self.y_train)
        y_pred = list(algorithm.predict(self.X_test))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc