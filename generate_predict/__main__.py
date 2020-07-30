"""
generates the master scripts
"""
# pylint: disable=invalid-name

import os

import argparse
from datetime import datetime, timedelta
description = \
    """
VoterLabs Inc.
creates predict script
VL uses the sliding window time series method for Univariate and Multivariate data and multi-step forecasting
Univariate Time Series: These are datasets where only a single variable is observed at each time, such as covid deaths per day each hour.
Multivariate Time Series: These are datasets where two or more variables are observed at each time.

  READ FILE_IN_RAW.CSV
  LOAD CSV DATA FROM YOUR COMPUTER

Must be a csv file where first row has column header names.
Must include time series date columns - like MM/DD/YY (7/3/20)
Must include targeted date or will automatically predict last date in series.
Must include as much data of cause of time series as you can - more data equals better predictions


  GET COLUMN HEADERS
  FOR EACH COLUMN NOT IN IGNORE LIST :
  GET ALL CATEGORIES = UNIQUE COLUMN VALUES
  GENERATE ONE HOT ENCODING for each HEADER COLUMN VALUE THAT IS CATEGORY BASED like CITY name
  ENCODE EACH ROW WITH 1 or 0 FOR EACH HEADER
  GENERATE discrete ENCODING for each HEADER COLUMN VALUE THAT IS CONTINUOUS BASED like sales
  Send the X,Y training and test splits of the encoded data into the models
  Then compare actual to predicted values


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
        '--target',
        help='column name to be targeted for prediction it can be categorical or continuous data')
    parser.add_argument(
        '--file_out_scores',
        help='output file scores the models - scores being accuracy, recall, precision - True Positive , '
             'False Positive, False Negative, True Negative for Confusion Matrix ')
    parser.add_argument(
        '--file_out_predict',
        help='predicted output versus actual')
    parser.add_argument(
        '--ignore',
        help='columns to not use in encoding or predictions')
    parser.add_argument(
        '--file_out_tableau',
        help='tableau formated prediction outputs: Location | Act |Pred | Date')
    parser.add_argument(
        '--file_in_master',
        help='raw csv file input to be mastered - Must be a csv file where first row has column header names.')

    parser.add_argument(
        '--file_in',
        help='raw csv file input to be predicted. Must be a csv file where first row has column header names. '
             'Must include time series date columns - like MM/DD/YY (7/3/20) ')
    parser.add_argument(
        '--predict_file_script_out',
        help='shell script for each time series sliced directory of data that creates predictions')
    parser.add_argument(
        '--start_date_all',
        help='start of time series window - each step is a day each column must be a date in format MM/DD/YY - like 7/3/20')
    parser.add_argument(
        '--end_date_all',
        help='end of time series window - each step is a day each column must be a date in format MM/DD/YY - like 7/22/20 ')
    parser.add_argument(
        '--window_size',
        help='number of time series increments per window - this is an integet like 4. '
             'This is the sliding window method for framing a time series dataset the increments are days')
    parser.add_argument(
        '--parent_dir',
        help='beginning of docker file system - like /app')

    parser.add_argument(
        '--add_model',
        action='append',
        help='names of the models to be tried - names of models are in description')
    args = parser.parse_args()
    return args


def main():
    """
    runs the predict module
    """
    # pylint: disable=invalid-name
    # pylint: disable=too-many-locals
    # pylint: disable=consider-using-sys-exit
    # pylint: disable=unused-variable
    # pylint: disable=too-many-statements
    args = parse_command_line()
    file_in = args.file_in
    target = args.target
    file_out_scores = args.file_out_scores
    file_in_master = args.file_in_master

    file_out_predict = args.file_out_predict
    start_date_all = args.start_date_all
    end_date_all = args.end_date_all
    predict_file_script_out = args.predict_file_script_out
    add_model = args.add_model
    ignore = args.ignore
    file_out_tableau = args.file_out_tableau

    window_size = args.window_size

    start_date_all_window_f = datetime.strptime(start_date_all, "%m/%d/%Y")
    end_date_all_window_f = datetime.strptime(end_date_all, "%m/%d/%Y")

    start_window_date_next = start_date_all_window_f
    end_window_date_next = start_date_all_window_f + \
        timedelta(days=int(window_size))
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

    while end_window_date_next < end_date_all_window_f:
        start_window_date = start_window_date_next
        end_window_date = end_window_date_next
        time_series = start_window_date.strftime(
            "%m-%d-%Y") + "_" + end_window_date.strftime("%m-%d-%Y")


        # Directory
        directory = time_series

        # Parent Directory path
        #parent_dir = "/Users/tomlorenc/Sites/VL_standard/ml"
        #parent_dir = "/app"

        # Path
        path = os.path.join(parent_dir, directory)

        tscsv = "_" + time_series + ".csv"
        file_out_predict_ts = file_out_predict.replace(".csv", tscsv)
        file_out_predict_ts_path = path + "/" + file_out_predict_ts

        file_out_scores_ts = file_out_scores.replace(".csv", tscsv)
        file_out_scores_path = path + "/" + file_out_scores_ts

        file_in_ts = file_in.replace(".csv", tscsv)
        file_in_ts_path = path + "/" + file_in_ts
        try:
            os.mkdir(path)
        except OSError as error:
            print("fick")

        start_date_window_f = start_window_date
        end_date_window_f = end_window_date

        models = []
        for model in add_model:
            models.append(model)

        model_options = "\\\n".join(f"  --predictor   {m}" for m in models)

        dates = [end_window_date]
        options = "\\\n".join(
            f"  --target   {d.strftime('%m/%d/%Y')}_DISCRETE" for d in dates)
        no = options.replace("/", r"\/")
        no2 = no.replace("2020", "20")
        no3 = no2.replace("03", "3")
        no4 = no3.replace("04", "4")
        no5 = no4.replace("05", "5")
        no6 = no5.replace("06", "6")
        no7 = no6.replace("07", "7")
        no8 = no7.replace("01", "1")
        no9 = no8.replace("02", "2")
        no10 = no9.replace("08", "8")
        no11 = no10.replace("09", "9")

        template = f"""
        #!/usr/bin/env bash
        python -m predict  \\
          --file_in {file_in_ts_path} \\
                    --file_in_master {file_in_master} \\
          --strategy none \\
          {no11} \\
         --ignore  {ignore} \\
          --training_test_split_percent 70 \\
             {model_options} \\
          --score f1_score \\
          --score classification_accuracy \\
          --score recall  \\
          --file_in_config config/ohe_config.yaml \\
          --file_out_scores {file_out_scores_path} \\
                    --file_out_scores {file_out_scores_path} \\
        --file_out_tableau {file_out_tableau} \\
          --file_out_predict {file_out_predict_ts_path}


        """.strip()

        print(template)

        discrete_text_file = open(predict_file_script_out, "w")

        discrete_text_file.write(template)

        tssh = "_" + time_series + ".sh"
        predict_file_script_out_ts = predict_file_script_out.replace(
            ".sh", tssh)
        predict_file_script_out_ts_path = path + "/" + predict_file_script_out_ts

        print("predict_file_script_out_ts_path ")
        print(predict_file_script_out_ts_path)
        print(template)
        discrete_text_file = open(predict_file_script_out_ts_path, "w")
        discrete_text_file.write(template)

        start_window_date_next = start_window_date_next + timedelta(days=1)
        end_window_date_next = start_window_date_next + \
            timedelta(days=int(window_size))


if __name__ == '__main__':
    main()
