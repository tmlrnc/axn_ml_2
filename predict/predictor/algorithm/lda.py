from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from predict.predictor import Commandline
from predict.predictor.algorithm.simple_predictor import SimplePredictor

model_name = 'Linear Discriminant Analysis'
algorithm = lambda: LinearDiscriminantAnalysis()

@Commandline("LDA")
class LDA_OHP(SimplePredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        """
         initializes the training and testing features and labels

         :param target: string - label to be predicted or classified
         :param X_test: array(float) - testing features
         :param X_train: array(float) - training features
         :param y_test: array(float) - testing label
         :param y_train: array(float) - testing label
         """
        super().__init__(target, X_test, X_train, y_test, y_train, model_name, algorithm)
