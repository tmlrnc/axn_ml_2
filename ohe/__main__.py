import argparse

from ohe.config import init_ohe_config
from ohe.predictor import OneHotPredictorBuilder, Runner, get_algorithm_from_string
from ohe.encoder import OneHotEncoderBuilder





def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in')
    parser.add_argument('--file_out_ohe')
    parser.add_argument('--file_out_predict')
    parser.add_argument('--training_test_split_percent', type=int)
    parser.add_argument('--file_in_config')
    parser.add_argument('--ohe_only')

    parser.add_argument(
        '--target',
        action='append')
    parser.add_argument(
        '--ignore',
        action='append')
    parser.add_argument(
        '--predictor',
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
    args = parse_command_line()
    file_in_name = args.file_in
    file_out_ohe = args.file_out_ohe
    file_out_predict = args.file_out_predict
    training_test_split_percent = args.training_test_split_percent
    file_in_config = args.file_in_config
    ohe_only = args.ohe_only

    init_ohe_config(file_in_config)

    algorithms = set( get_algorithm_from_string(p.strip().upper()) for p in args.predictor)
    ohe_builder = OneHotEncoderBuilder(file_in_name)


    for ignore in args.ignore:
        ohe_builder.ignore(ignore)
    ohe = ohe_builder.build()
    data_frame, feature_name_list = ohe.one_hot_encode()
    ohe.write_ohe_csv(file_out_ohe)
    all_targets = []
    for new_target in args.target:
        if new_target == "All":
            all_targets = (f for f in feature_name_list if f not in args.ignore)
            break
        else:
            all_targets.append(new_target)
    # build across all targets
    # get all target names
    write_header_flag = 1
    if ohe_only != "YES":
        for new_target in all_targets:
            ohp_builder = OneHotPredictorBuilder(new_target, training_test_split_percent, data_frame)
            features = (f for f in feature_name_list if f not in args.ignore and f != new_target)
            for f in features:
                ohp_builder.add_feature(f)
            runner = Runner(ohp_builder, algorithms)
            runner.run_and_build_predictions()
            runner.write_predict_csv(file_out_predict, new_target,write_header_flag)





if __name__ == '__main__':
    main()


