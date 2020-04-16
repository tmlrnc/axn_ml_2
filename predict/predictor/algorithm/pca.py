from sklearn.decomposition import PCA

from predict.predictor import OneHotPredictor, Commandline
from predict.config import get_ohe_config

@Commandline("PCA")
class PCA_OHP(OneHotPredictor):

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
        self.model_name = 'PCA'

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

        algorithm = PCA(n_components=2, svd_solver='full')
        _explained_variance_ = algorithm.explained_variance_
        _explained_variance_ratio_ = algorithm.explained_variance_ratio_
        _covariance = algorithm.get_covariance()

        print(_covariance)



        #algorithm.fit(self.X_train.toarray(), self.y_train)
        #y_pred = list(algorithm.predict(self.X_test.toarray()))
        #self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return 0