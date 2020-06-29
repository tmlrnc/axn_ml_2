import argparse

from predict.config import init_ohe_config
import csv

import pandas
import pandas as pd
import numpy

from predict.predictor import OneHotPredictorBuilder, Runner, get_algorithm_from_string


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in')
    parser.add_argument('--file_in_master')
    parser.add_argument('--file_out_predict')
    parser.add_argument('--file_out_tableau')
    parser.add_argument('--file_out_scores')
    parser.add_argument('--strategy')
    parser.add_argument('--training_test_split_percent', type=int)
    parser.add_argument('--file_in_config')
    parser.add_argument('--target')
    parser.add_argument(
        '--predictor',
        action='append')
    parser.add_argument(
        '--score',
        action='append')
    parser.add_argument(
        '--ignore',
        action='append')
    args = parser.parse_args()
    return args


def main():
    """
  READ FILE_IN_RAW.CSV
  GET COLUMN HEADERS
  FOR EACH COLUMN NOT IN IGNORE LIST :
  GET ALL CATEGORIES = UNIQUE COLUMN VALUES
  GENERATE ONE HOT ENCODING HEADER
  ENCODE EACH ROW WITH 1 or 0 FOR EACH HEADER


"RFR" RandomForestRegressor - random_forest_regression.py
"LR" LogisticRegression - logistic_regression.py
"MLP" MLPRegressor - mlp_regression.py
"SVM" LinearSVC - svm.py
"NUSVM" NuSVC - nu_svm.py
"BFRA" BruteForceRadius - brute_force_radius.py
"NUSVMSIG" NU_SVM_Sigmoid - nu_svm_sigmoid.py
"LSQLDA" LeastSquaresLDA - least_sqaures_LDA.py
"MULTICLASSLR" MULTI_CLASS_LogisticRegression - multi_class_logistic_regression.py
"RIDGEREGRESSION" RIDGE_REGRESSION - ridge_regression.py
"LASSOMODEL" Lasso - lasso.py
“BAYESIANRIDGE” BayesianRidge - bayesian_ridge.py
KNeighborsRegressor - kneighborregressor.py
Kmeans - kmeans.py
LARSLASSOR LassoLars - lars_lasso.py
LSQLDA - least_sqaures_LDA.py
LINEARREGRESSION LinearRegression - linear_regression.py
NONLINSVM non linear svm linear_svm.py
NONLINSVMSIGMOID SVC(kernel='sigmoid') - linear_svm_sigmoid.py
PERCEPTRONNEURALNET - Perceptron - perceptron_neural_net.py
PERCEPTRONNEURALNETNONLINEAR - perceptron_neural_net_non_linear.py
PERCEPTRONNEURALNETNONLINEARL1 - perceptron_neural_net_l1_penalty.py
PERCEPTRONNEURALNETNONLINEARELASTIC - perceptron_neural_net_elastic_net_penalty.py
RIDGECROSSVALIDATION RidgeCV - ridge_cross_validation.py
RIDGECROSSVALIDATIONNORM ridge_cross_validation_normalized.py
KMEDIAN - kmedian.py
RANDOMFORRESTREGMAX - RandomForestClassifier - random_forest_classifier_max.py
PACCLASSWE - PassiveAggressiveClassifier weighted -
ENTROPY_DECISION_TREE - EDT - entropy_decision_tree.py
RANDOMDT - random_decision_tree.py
KNeighborsClassifier - KNCLASS - knclass.py
KNCLASSMINKOW - Minkowski Tree - knnclassminkow.py
MLPNESTEROV - mlp_nesterov.py
OrthogonalMatchingPursuit - omp.py
PASSAGGCLASS - pac.py
PASSAGGCLASSEARLY - pacearly.py
RNRREG - rnr.py
DISTRNRREG - distancernr.py
QDANN - qda.py
MLPCLASSALPHA
GNB - Gaussian Niave Bayes gnb.py
ETC - ExtraTreesClassifier etc.py
ELNLRREG - elasticnet - elastic_net_logistic_regression.py
GNBAYESSMOOTHING - gnb_smoothing.py





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
    algorithms = set( get_algorithm_from_string(p.strip().upper()) for p in args.predictor)
    ######################################################################

    print("Predict Scoring --- START ")

    #
    # Predict Scoring
    #
    #data_frame = pandas.read_csv(file_in_name).fillna(value=0)


    print("self.ignore_list " + str(ignore_list))
    data_frame_all = pandas.read_csv(file_in_name).fillna(value=0)

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
    ohp_builder = OneHotPredictorBuilder(new_target, training_test_split_percent, data_frame_org, strategy)
    features = [f for f in feature_name_list if f != new_target]
    print("features: " + str(list(features)))

    for f in features:
        ohp_builder.add_feature(f)
    runner = Runner(ohp_builder, algorithms)
    runner.run_and_build_predictions(score_list,file_out_predict,  data_frame_ignore_frame, training_test_split_percent, ignore_list, write_actual_flag)
    runner.write_predict_csv(file_out_scores, file_out_tableau, file_in_master,file_out_predict, new_target,write_header_flag)


    print("Predict Scoring --- END ")


if __name__ == '__main__':
    main()


