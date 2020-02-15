from sklearn.neighbors import KNeighborsClassifier
from ohe.predictor import OneHotPredictor, Commandline
from ohe.config import get_ohe_config

@Commandline("KNN")
class KNN_OHP(OneHotPredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        super().__init__(target, X_test, X_train, y_test, y_train)
        self.model_name = 'K Nearest Neighbors'

    def predict(self):

        algorithm = KNeighborsClassifier(n_neighbors=3)
        algorithm.fit(self.X_train.toarray(), self.y_train)
        y_pred = list(algorithm.predict(self.X_test.toarray()))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc