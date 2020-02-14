from sklearn.ensemble import RandomForestClassifier

from ohe.predictor import OneHotPredictor, Commandline
from ohe.config import get_ohe_config

@Commandline("RF")
class RandomForest_OHP(OneHotPredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        super().__init__(target, X_test, X_train, y_test, y_train)
        self.model_name = 'Random Forest'

    def predict(self):
        algorithm = RandomForestClassifier(n_estimators=get_ohe_config().RF_n_estimators,
                                        max_depth=get_ohe_config().RF_max_depth)
        algorithm.fit(self.X_train, self.y_train)
        y_pred = list(algorithm.predict(self.X_test))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc