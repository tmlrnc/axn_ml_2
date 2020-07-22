from sklearn.neural_network import MLPClassifier

from predict.predictor import OneHotPredictor, Commandline
from predict.config import get_ohe_config

@Commandline("MLPCLASS2")
class MLPClassifier2_OHP(OneHotPredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        super().__init__(target, X_test, X_train, y_test, y_train)
        self.model_name = 'MLPCLASS2'

    def predict(self):
        algorithm = MLPClassifier(solver=get_ohe_config().mlp_solver, alpha=get_ohe_config().mlp_alpha,
                                 hidden_layer_sizes=(get_ohe_config().mlp_layers, get_ohe_config().mlp_neurons),
                                 random_state=get_ohe_config().mlp_random_state)
        algorithm.fit(self.X_train, self.y_train)
        y_pred = list(algorithm.predict(self.X_test))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc