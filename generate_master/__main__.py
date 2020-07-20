from covid import downloader
import datetime as dt
import argparse
from datetime import datetime, timedelta
import requests
import filecmp
import logging
import os
import subprocess
from subprocess import PIPE, Popen

description = \
"""
VoterLabs Inc. 
Master Control

  LOAD CSV DATA FROM YOUR COMPUTER 

Must be a csv file where first row has column header names. 
Must include time series date columns - MM/DD/YY (7/3/20)
Must include targeted date or will automatically predict last date in series.
Must include as much data of cause of time series as you can - more data equals better predictions 

creates all scripts and excecutes them

  READ FILE_IN_RAW.CSV
  GET COLUMN HEADERS
  FOR EACH COLUMN NOT IN IGNORE LIST :
  GET ALL CATEGORIES = UNIQUE COLUMN VALUES
  GENERATE ONE HOT ENCODING HEADER
  ENCODE EACH ROW WITH 1 or 0 FOR EACH HEADER
  
  Then split into test and training sets such that:
  Training data set—a subset to train a model.
  Test data set—a subset to test the trained model.
  Test set MUST meet the following two conditions:
  Is large enough to yield statistically meaningful results.
  Is representative of the data set as a whole. 
  Don't pick a test set with different characteristics than the training set.
  Then we train models using Supervised learning.
  Supervised learning consists in learning the link between two datasets: 
  the observed data X and an external variable y that we are trying to predict, called “target”
  Y is a 1D array of length n_samples.
  All VL models use a fit(X, y) method to fit the model and a predict(X) method that, given unlabeled observations X, returns the predicted target y.

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
    parser = argparse.ArgumentParser(description=description)


    parser.add_argument('--file_in', help='raw csv file input to be predicted. Must be a csv file where first row has column header names. Must include time series date columns - like MM/DD/YY (7/3/20) ')
    parser.add_argument('--master_file_script_out', help='master shell script for full automation')
    parser.add_argument('--ohe_file_script_out', help='shell script for one hot encoding')
    parser.add_argument('--predict_file_script_out', help='shell script for prediction')
    parser.add_argument('--discrete_file_script_out', help='shell script for one hot discretized')

    parser.add_argument('--start_date_all', help='start of time series window - each step is a day each column must be a date in format MM/DD/YY - like 7/3/20')
    parser.add_argument('--end_date_all', help='end of time series window - each step is a day each column must be a date in format MM/DD/YY - like 7/22/20 ')
    parser.add_argument('--window_size', help='number of time series increments per window - this is an integet like 4. This is the sliding window method for framing a time series dataset the increments are days')
    parser.add_argument('--parent_dir', help='beginning of docker file system - like /app')


    args = parser.parse_args()
    return args



def main():
    log = logging.getLogger("logger")
    log.setLevel(logging.INFO)
    logging.basicConfig()

    log.info("IM MASTER")
    args = parse_command_line()
    file_in = args.file_in
    start_date_all = args.start_date_all
    end_date_all = args.end_date_all
    discrete_file_script_out = args.discrete_file_script_out

    master_file_script_out = args.master_file_script_out
    predict_file_script_out = args.predict_file_script_out

    window_size = args.window_size

    ohe_file_script_out = args.ohe_file_script_out

    start_date_all_window_f = datetime.strptime(start_date_all, "%m/%d/%Y")
    end_date_all_window_f = datetime.strptime(end_date_all, "%m/%d/%Y")

    start_window_date_next = start_date_all_window_f
    end_window_date_next = start_date_all_window_f + timedelta(days=int(window_size))
    print("start_window_date_next ")
    print(start_window_date_next)
    print("end_window_date_next ")
    print(end_window_date_next)
    print(end_date_all_window_f)

    parent_dir = args.parent_dir
    if parent_dir is None:
        print("Parent dir is not specified.")
        quit()
    print(f"Using parent_dir: {parent_dir}")

    while (end_window_date_next < end_date_all_window_f):
        start_window_date = start_window_date_next
        end_window_date = end_window_date_next
        time_series = start_window_date.strftime("%m-%d-%Y") + "_" + end_window_date.strftime("%m-%d-%Y")

        import os

        # Directory
        directory = time_series

        # Parent Directory path
        #parent_dir = "/Users/tomlorenc/Sites/VL_standard/ml"
        #parent_dir = "/app"


        # Path
        path = os.path.join(parent_dir, directory)

        tssh = "_" + time_series + ".sh"
        discrete_file_script_out_ts = discrete_file_script_out.replace(".sh", tssh)
        discrete_file_script_out_ts_path = path + "/" + discrete_file_script_out_ts

        tssh = "_" + time_series + ".sh"
        ohe_file_script_out_ts = ohe_file_script_out.replace(".sh", tssh)
        ohe_file_script_out_ts_path = path + "/" + ohe_file_script_out_ts

        tssh = "_" + time_series + ".sh"
        predict_file_script_out_ts = predict_file_script_out.replace(".sh", tssh)
        predict_file_script_out_ts_path = path + "/" + predict_file_script_out_ts



        try:
            os.mkdir(path)
        except OSError as error:
            print("fick")

        start_date_window_f = start_window_date
        end_date_window_f = end_window_date
        import time
        start = time.time()
        import os

        comm = "exec bash " + discrete_file_script_out_ts_path
        os.system(comm)
        comm2 = "exec bash " + ohe_file_script_out_ts_path
        print(comm2)

        os.system(comm2)

        comm3 = "bash " + predict_file_script_out_ts_path
        log.info("IM MASTER")

        print(comm3)
        os.system(comm3)

        #p = subprocess.Popen(['bash', predict_file_script_out_ts_path], stdin=PIPE, stdout=PIPE)
        #one_line_output2 = p2.stdout.readline()
        #print(one_line_output2)
        #log.info("IM MASTER" + str(predict_file_script_out_ts_path))
        #log.info("IM MASTER" + str(one_line_output2))


        print('It took {0:0.1f} seconds'.format(time.time() - start))


        start_window_date_next = start_window_date_next + timedelta(days=1)
        end_window_date_next = start_window_date_next + timedelta(days=int(window_size))


if __name__ == '__main__':
    main()


