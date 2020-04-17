from sklearn.neural_network import MLPClassifier

from predict.predictor import OneHotPredictor, Commandline
from predict.config import get_ohe_config

@Commandline("MLPCLASSALPHA")
class MLPClassifierAlpha_OHP(OneHotPredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        super().__init__(target, X_test, X_train, y_test, y_train)
        self.model_name = 'MLPCLASSALPHA'

    def predict(self):
        algorithm = MLPClassifier(solver=get_ohe_config().MLP_solver, alpha=get_ohe_config().MLP_alpha,
                                 hidden_layer_sizes=(get_ohe_config().MLP_layers, get_ohe_config().MLP_neurons),
                                 random_state=get_ohe_config().MLP_random_state)
        algorithm.fit(self.X_train, self.y_train)
        y_pred = list(algorithm.predict(self.X_test))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc