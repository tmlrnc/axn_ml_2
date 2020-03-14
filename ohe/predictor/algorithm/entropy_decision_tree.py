from sklearn.tree import DecisionTreeClassifier
from ohe.predictor import OneHotPredictor, Commandline
from ohe.config import get_ohe_config

@Commandline("ENTROPY_DECISION_TREE")
class ENTROPY_DECISION_TREE_OHP(OneHotPredictor):

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
        self.model_name = 'ENTROPY_DECISION_TREE'

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
        algorithm = DecisionTreeClassifier(criterion='entropy')
        algorithm.fit(self.X_train.toarray(), self.y_train)
        y_pred = list(algorithm.predict(self.X_test.toarray()))
        F1 = OneHotPredictor.get_f1_score(y_pred, self.y_test)


        print("self.F1 score " + str(F1))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)

        return self.acc