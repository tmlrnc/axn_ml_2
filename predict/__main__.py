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
    parser.add_argument('--file_out')
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

      """
    ######################################################################
    #
    # read run commands
    #
    args = parse_command_line()
    file_in_name = args.file_in
    file_out = args.file_out
    training_test_split_percent = args.training_test_split_percent
    file_in_config = args.file_in_config
    strategy = args.strategy
    score_list = args.score

    init_ohe_config(file_in_config)
    new_target = args.target
    algorithms = set( get_algorithm_from_string(p.strip().upper()) for p in args.predictor)
    ######################################################################

    print("Predict Scoring --- START ")

    #
    # Predict Scoring
    #
    data_frame = pandas.read_csv(file_in_name).fillna(value=0)
    feature_name_list = list(data_frame.columns)
    write_header_flag = 1
    ohp_builder = OneHotPredictorBuilder(new_target, training_test_split_percent, data_frame, strategy)
    features = [f for f in feature_name_list if f != new_target]
    print("features: " + str(list(features)))

    for f in features:
        ohp_builder.add_feature(f)
    runner = Runner(ohp_builder, algorithms)
    runner.run_and_build_predictions(score_list)
    runner.write_predict_csv(file_out, new_target,write_header_flag)


    print("Predict Scoring --- END ")


if __name__ == '__main__':
    main()


