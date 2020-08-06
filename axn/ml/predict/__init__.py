"""

Predict Module

<img src="images/model.png" alt="DIS">

Step 1
----------
    READ FILE_IN_RAW.CSV
    reads config yaml file into config dictionary data object
    to drive the scikit learn machine learning algorithm


Step 2
----------
    GET COLUMN HEADERS



Step 3
----------
    FOR EACH COLUMN NOT IN IGNORE LIST




Step 4
----------
    GET ALL CATEGORIES = UNIQUE COLUMN VALUES



Step 5
----------
    GET COLUMN HEADERS



Step 6
----------
    DISCRTTIZE or OHE EACH ROW INTO BIN



Step 7
----------
    Then split into test and training sets such that: Training data set—a subset to train a model.
    Test data set—a subset to test the trained model.



Step 8
----------
    Then we train models using Supervised learning.



Step 9
----------
    The observed data X and an external variable y that we are trying to predict, called “target”
    Y is a 1D array of length n_samples.
    All VL models use a fit(X, y) method to fit the model and a predict(X) method that,
    given unlabeled observations X, returns the predicted labels y.
    a regressor model is a set of methods intended for regression in which the
    target value is expected to be a linear combination of the features.




Step 10
----------

    Model doc definition

    "RFR"                             RandomForestRegressor           -          random_forest_regression.py
      ^                                     ^                                              ^
      ^                                     ^                                              ^
      ^                                     ^                                              ^
    initials of the model              full name of model                        file name of model
    to be added to
    predictor parameter


    "RFR" RandomForestRegressor - random_forest_regression.py
    "LR" LogisticRegression - logistic_regression.py
    "MLP" MLP Regressor - mlp_regression.py
    "SVM" Linear SVC - svm.py
    "NUSVM" Nu SVC - nu_svm.py
    "BFRA" Brute Force Radius - brute_force_radius.py
    "NUSVMSIG" NU SVM sigmoid - nu_svm_sigmoid.py
    "LSQLDA" Least Squares LDA - least_sqaures_LDA.py
    "MULTICLASSLR" MULTI CLASS_Logistic Regression - multi_class_logistic_regression.py
    "RIDGEREGRESSION" RIDGE REGRESSION - ridge_regression.py
    "LASSOMODEL" Lasso - lasso.py
    “BAYESIANRIDGE” BayesianRidge - bayesian_ridge.py
    "KNeighborsRegressor" KNeighborsRegressor - kneighborregressor.py
    "Kmeans" Kmeans - kmeans.py
    "LARSLASSOR" LassoLars - lars_lasso.py
    "LSQLDA" least_sqaures_LDA.py
    "LINEARREGRESSION" Linear Regression - linear_regression.py
    "NONLINSVM" non linear svm - linear_svm.py
    "NONLINSVMSIGMOID" SVC(kernel='sigmoid') - linear_svm_sigmoid.py
    "PERCEPTRONNEURALNET"  Perceptron - perceptron_neural_net.py
    "PERCEPTRONNEURALNETNONLINEAR" Perceptron Non Linear - perceptron_neural_net_non_linear.py
    "PERCEPTRONNEURALNETNONLINEARL1"  Perceptron Non Linear 1 - perceptron_neural_net_l1_penalty.py
    "PERCEPTRONNEURALNETNONLINEARELASTIC - perceptron_neural_net_elastic_net_penalty.py
    "RIDGECROSSVALIDATION" RidgeCV - ridge_cross_validation.py
    "RIDGECROSSVALIDATIONNORM" RidgeCV Norm -  ridge_cross_validation_normalized.py
    "KMEDIAN"  KMEDIAN - kmedian.py
    "RANDOMFORRESTREGMAX"  RandomForestClassifier - random_forest_classifier_max.py
    "PACCLASSWE"  PassiveAggressiveClassifier weighted - passiveaggressiveclassifier.py
    "ENTROPY_DECISION_TREE" EDT - entropy_decision_tree.py
    "RANDOMDT" random decision tree - random_decision_tree.py
    "KNeighborsClassifier"  KNCLASS - knclass.py
    "KNCLASSMINKOW"  Minkowski Tree - knnclassminkow.py
    "MLPNESTEROV" mlp nesterov - mlp_nesterov.py
    "OrthogonalMatchingPursuit" Orthogonal Matching Pursuit - omp.py
    "PASSAGGCLASS" PASSIVE AGGREESISVE CLASSIFICATION - pac.py
    "PASSAGGCLASSEARLY" PASSIVE AGGREESISVE CLASSIFICATION  - pacearly.py
    "RNRREG" regional regression - rnr.py
    "DISTRNRREG" distance regression - distancernr.py
    "QDANN" quadradic aggressor - qda.py
    "MLPCLASSALPHA" mlp aggressor - mlpapha.py
    "GNB" - Gaussian Niave Bayes - gnb.py
    "ETC" - ExtraTreesClassifier - etc.py
    "ELNLRREG" - elastic net - elastic_net_logistic_regression.py
    "GNBAYESSMOOTHING" guassian niave bayes smoothing - gnb_smoothing.py



Step 11
----------
    Choose most accurate model



Returns:
----------
    csv file output with predicted results vs actual results


Example 1. CSV Files:
---------------------
python -m discrete  \


  --file_in csvs/sales.csv \




  --file_out_ohe csvs/sales_dis.csv \


  --ignore id


  --ignore item



Example 1 - Data Input CSV File:
----------------------------
<img src="images/sales.png" alt="OHE" width="600" height="300">


Example 1 - One Hot Encoded CSV File:
-----------------------------
<img src="images/sales_dis.png" alt="OHE" width="600" height="300">



"""
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=trailing-whitespace
# pylint: disable=too-many-statements
# pylint: disable=line-too-long


import argparse
import pandas as pd
from .config import init_ohe_config
from .predictor import OneHotPredictorBuilder, Runner, get_algorithm_from_string

description = \
    """
        VoterLabs Inc.




  Then split into test and training sets such that: Training data set—a subset to train a model. 
  Test data set—a subset to test the trained model.
  Test set MUST meet the following two conditions:
  Is large enough to yield statistically meaningful results.
  Is representative of the data set as a whole.
  Don't pick a test set with different characteristics than the training set.
  Then we train models using Supervised learning.
  Supervised learning consists in learning the link between two datasets:
  the observed data X and an external variable y that we are trying to predict, called “target”
  Y is a 1D array of length n_samples.
  All VL models use a fit(X, y) method to fit the model and a predict(X) method that, 
  given unlabeled observations X, returns the predicted labels y.
  
  a regressor model is a set of methods intended for regression in which the target 
  value is expected to be a linear combination of the features.

Model doc definition

"RFR"                             RandomForestRegressor           -          random_forest_regression.py
  ^                                     ^                                              ^
  ^                                     ^                                              ^
  ^                                     ^                                              ^
initials of the model              full name of model                        file name of model
to be added to
predictor parameter


"RFR" RandomForestRegressor - random_forest_regression.py
"LR" LogisticRegression - logistic_regression.py
"MLP" MLP Regressor - mlp_regression.py
"SVM" Linear SVC - svm.py
"NUSVM" Nu SVC - nu_svm.py
"BFRA" Brute Force Radius - brute_force_radius.py
"NUSVMSIG" NU SVM sigmoid - nu_svm_sigmoid.py
"LSQLDA" Least Squares LDA - least_sqaures_LDA.py
"MULTICLASSLR" MULTI CLASS_Logistic Regression - multi_class_logistic_regression.py
"RIDGEREGRESSION" RIDGE REGRESSION - ridge_regression.py
"LASSOMODEL" Lasso - lasso.py
“BAYESIANRIDGE” BayesianRidge - bayesian_ridge.py
"KNeighborsRegressor" KNeighborsRegressor - kneighborregressor.py
"Kmeans" Kmeans - kmeans.py
"LARSLASSOR" LassoLars - lars_lasso.py
"LSQLDA" least_sqaures_LDA.py
"LINEARREGRESSION" Linear Regression - linear_regression.py
"NONLINSVM" non linear svm - linear_svm.py
"NONLINSVMSIGMOID" SVC(kernel='sigmoid') - linear_svm_sigmoid.py
"PERCEPTRONNEURALNET"  Perceptron - perceptron_neural_net.py
"PERCEPTRONNEURALNETNONLINEAR" Perceptron Non Linear - perceptron_neural_net_non_linear.py
"PERCEPTRONNEURALNETNONLINEARL1"  Perceptron Non Linear 1 - perceptron_neural_net_l1_penalty.py
"PERCEPTRONNEURALNETNONLINEARELASTIC - perceptron_neural_net_elastic_net_penalty.py
"RIDGECROSSVALIDATION" RidgeCV - ridge_cross_validation.py
"RIDGECROSSVALIDATIONNORM" RidgeCV Norm -  ridge_cross_validation_normalized.py
"KMEDIAN"  KMEDIAN - kmedian.py
"RANDOMFORRESTREGMAX"  RandomForestClassifier - random_forest_classifier_max.py
"PACCLASSWE"  PassiveAggressiveClassifier weighted - passiveaggressiveclassifier.py
"ENTROPY_DECISION_TREE" EDT - entropy_decision_tree.py
"RANDOMDT" random decision tree - random_decision_tree.py
"KNeighborsClassifier"  KNCLASS - knclass.py
"KNCLASSMINKOW"  Minkowski Tree - knnclassminkow.py
"MLPNESTEROV" mlp nesterov - mlp_nesterov.py
"OrthogonalMatchingPursuit" Orthogonal Matching Pursuit - omp.py
"PASSAGGCLASS" PASSIVE AGGREESISVE CLASSIFICATION - pac.py
"PASSAGGCLASSEARLY" PASSIVE AGGREESISVE CLASSIFICATION  - pacearly.py
"RNRREG" regional regression - rnr.py
"DISTRNRREG" distance regression - distancernr.py
"QDANN" quadradic aggressor - qda.py
"MLPCLASSALPHA" mlp aggressor - mlpapha.py
"GNB" - Gaussian Niave Bayes - gnb.py
"ETC" - ExtraTreesClassifier - etc.py
"ELNLRREG" - elastic net - elastic_net_logistic_regression.py
"GNBAYESSMOOTHING" guassian niave bayes smoothing - gnb_smoothing.py




              """.strip()


def parse_command_line():
    """
    reads the command line args
    """
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--file_in',
        help='raw csv file input to be predicted. Must be a csv file where first row has column header names. '
             'Must include time series date columns - like MM/DD/YY (7/3/20) ')
    parser.add_argument(
        '--file_in_master',
        help='raw csv file input to be mastered')
    parser.add_argument(
        '--file_out_predict',
        help='predicted output verus actual')
    parser.add_argument(
        '--file_out_tableau',
        help='tableau formated prediction outputs: Location | Act |Pred | Date')
    parser.add_argument('--file_out_scores', help='tabelau formteed')
    parser.add_argument('--strategy', help='predict startegy')
    parser.add_argument(
        '--training_test_split_percent',
        type=int,
        help='Then split into test and training sets such that: Training data set—a subset to train a model. '
             'Test data set—a subset to test the trained model.')
    parser.add_argument(
        '--file_in_config',
        help='model config that holds the hyperparametrs of algorthms like number of iterations')
    parser.add_argument(
        '--target',
        help='column name to be targeted for prediction it can be categorical or continuous data')
    parser.add_argument(
        '--predictor',
        action='append',
        help='the initals of the machine learning algortihm you wish to include in the optimization process - the initals are in the description')
    parser.add_argument(
        '--score',
        action='append',
        help='output file scores the models - scores being accuracy, recall, precision - True Positive , '
             'False Positive, False Negative, True Negative for Confusion Matrix ')
    parser.add_argument(
        '--ignore',
        action='append',
        help='columns of data to NOT be encoded or discretized - remove from processing without '
             'removing from raw data because they might be usseful to know latrer - like first name')
    args = parser.parse_args()
    return args


def main():
    """

    <img src="images/ml.png" alt="DIS">



 Step 1
----------
    READ FILE_IN_RAW.CSV
    reads config yaml file into config dictionary data object
    to drive the scikit learn machine learning algorithm


Step 2
----------
    GET COLUMN HEADERS



Step 3
----------
    FOR EACH COLUMN NOT IN IGNORE LIST




Step 4
----------
    GET ALL CATEGORIES = UNIQUE COLUMN VALUES



Step 5
----------
    GET COLUMN HEADERS



Step 6
----------
    DISCRETIZE or OHE EACH ROW INTO BIN



Step 7
----------
    Then split into test and training sets such that: Training data set—a subset to train a model.
    Test data set—a subset to test the trained model.



Step 8
----------
    Then we train models using Supervised learning.



Step 9
----------
    The observed data X and an external variable y that we are trying to predict, called “target”
    Y is a 1D array of length n_samples.
    All VL models use a fit(X, y) method to fit the model and a predict(X) method that,
    given unlabeled observations X, returns the predicted labels y.
    a regressor model is a set of methods intended for regression in which the target value is
    expected to be a linear combination of the features.




Step 10
----------

    Model doc definition

    "RFR"                             RandomForestRegressor           -          random_forest_regression.py
      ^                                     ^                                              ^
      ^                                     ^                                              ^
      ^                                     ^                                              ^
    initials of the model              full name of model                        file name of model
    to be added to
    predictor parameter


    "RFR" RandomForestRegressor - random_forest_regression.py
    "LR" LogisticRegression - logistic_regression.py
    "MLP" MLP Regressor - mlp_regression.py
    "SVM" Linear SVC - svm.py
    "NUSVM" Nu SVC - nu_svm.py
    "BFRA" Brute Force Radius - brute_force_radius.py
    "NUSVMSIG" NU SVM sigmoid - nu_svm_sigmoid.py
    "LSQLDA" Least Squares LDA - least_sqaures_LDA.py
    "MULTICLASSLR" MULTI CLASS_Logistic Regression - multi_class_logistic_regression.py
    "RIDGEREGRESSION" RIDGE REGRESSION - ridge_regression.py
    "LASSOMODEL" Lasso - lasso.py
    “BAYESIANRIDGE” BayesianRidge - bayesian_ridge.py
    "KNeighborsRegressor" KNeighborsRegressor - kneighborregressor.py
    "Kmeans" Kmeans - kmeans.py
    "LARSLASSOR" LassoLars - lars_lasso.py
    "LSQLDA" least_sqaures_LDA.py
    "LINEARREGRESSION" Linear Regression - linear_regression.py
    "NONLINSVM" non linear svm - linear_svm.py
    "NONLINSVMSIGMOID" SVC(kernel='sigmoid') - linear_svm_sigmoid.py
    "PERCEPTRONNEURALNET"  Perceptron - perceptron_neural_net.py
    "PERCEPTRONNEURALNETNONLINEAR" Perceptron Non Linear - perceptron_neural_net_non_linear.py
    "PERCEPTRONNEURALNETNONLINEARL1"  Perceptron Non Linear 1 - perceptron_neural_net_l1_penalty.py
    "PERCEPTRONNEURALNETNONLINEARELASTIC - perceptron_neural_net_elastic_net_penalty.py
    "RIDGECROSSVALIDATION" RidgeCV - ridge_cross_validation.py
    "RIDGECROSSVALIDATIONNORM" RidgeCV Norm -  ridge_cross_validation_normalized.py
    "KMEDIAN"  KMEDIAN - kmedian.py
    "RANDOMFORRESTREGMAX"  RandomForestClassifier - random_forest_classifier_max.py
    "PACCLASSWE"  PassiveAggressiveClassifier weighted - passiveaggressiveclassifier.py
    "ENTROPY_DECISION_TREE" EDT - entropy_decision_tree.py
    "RANDOMDT" random decision tree - random_decision_tree.py
    "KNeighborsClassifier"  KNCLASS - knclass.py
    "KNCLASSMINKOW"  Minkowski Tree - knnclassminkow.py
    "MLPNESTEROV" mlp nesterov - mlp_nesterov.py
    "OrthogonalMatchingPursuit" Orthogonal Matching Pursuit - omp.py
    "PASSAGGCLASS" PASSIVE AGGREESISVE CLASSIFICATION - pac.py
    "PASSAGGCLASSEARLY" PASSIVE AGGREESISVE CLASSIFICATION  - pacearly.py
    "RNRREG" regional regression - rnr.py
    "DISTRNRREG" distance regression - distancernr.py
    "QDANN" quadradic aggressor - qda.py
    "MLPCLASSALPHA" mlp aggressor - mlpapha.py
    "GNB" - Gaussian Niave Bayes - gnb.py
    "ETC" - ExtraTreesClassifier - etc.py
    "ELNLRREG" - elastic net - elastic_net_logistic_regression.py
    "GNBAYESSMOOTHING" guassian niave bayes smoothing - gnb_smoothing.py



Step 11
----------
    Choose most accurate model



Returns:
----------
    csv file output with predicted results vs actual results




      """
    ######################################################################
    #
    # read run commands
    #

    args = parse_command_line()
    file_in_name = args.file_in
    file_in_master = args.file_in_master

    file_out_tableau = args.file_out_tableau
    file_out_scores = args.file_out_scores
    file_out_predict = args.file_out_predict

    training_test_split_percent = args.training_test_split_percent
    file_in_config = args.file_in_config
    strategy = args.strategy
    score_list = args.score
    ignore_list = []
    for ignore in args.ignore:
        ignore_list.append(ignore)
    init_ohe_config(file_in_config)
    new_target = args.target
    algorithms = set(get_algorithm_from_string(p.strip().upper())
                     for p in args.predictor)
    ######################################################################

    print("Predict Scoring --- START ")

    #
    # Predict Scoring
    #
    #data_frame = pandas.read_csv(file_in_name).fillna(value=0)

    print("self.ignore_list " + str(ignore_list))
    data_frame_all = pd.read_csv(file_in_name).fillna(value=0)

    data_frame_org = data_frame_all

    data_frame_org = data_frame_all.drop(ignore_list, 1)

    data_frame_ignore_frame = data_frame_all[ignore_list]
    # print("self.data_frame_ignore_frame " + str(self.data_frame_ignore_frame))

    print("self.data_frame_ignore_frame ")

    print(data_frame_ignore_frame)

    # print("self.data_frame_ignore_frame_list " + str(self.data_frame_ignore_frame_list))

    feature_name_list = list(data_frame_org.columns)
    write_header_flag = 1
    write_actual_flag = 1
    ohp_builder = OneHotPredictorBuilder(
        new_target,
        training_test_split_percent,
        data_frame_org,
        strategy)
    features = [f for f in feature_name_list if f != new_target]
    print("features: " + str(list(features)))

    for f in features:
        ohp_builder.add_feature(f)
    runner = Runner(ohp_builder, algorithms)
    runner.run_and_build_predictions(
        score_list,
        file_out_predict,
        data_frame_ignore_frame,
        training_test_split_percent,
        ignore_list,
        write_actual_flag)
    runner.write_predict_csv(
        file_out_scores,
        file_out_tableau,
        file_in_master,
        file_out_predict,
        new_target,
        write_header_flag)

    print("Predict Scoring --- END ")
