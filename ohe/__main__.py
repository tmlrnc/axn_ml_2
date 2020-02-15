import argparse

from ohe.config import init_ohe_config
from ohe.predictor import OneHotPredictorBuilder, Runner, get_algorithm_from_string
from ohe.encoder import OneHotEncoderBuilder

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in')
    parser.add_argument('--file_out_ohe')
    parser.add_argument('--file_out_predict')
    parser.add_argument('--target')
    parser.add_argument('--training_test_split_percent', type=int)
    parser.add_argument('--file_in_config')

    parser.add_argument(
        '--ignore',
        action='append')
    parser.add_argument(
        '--predictor',
        action='append')
    args = parser.parse_args()
    return args

def main():
    args = parse_command_line()
    file_in_name = args.file_in
    file_out_ohe = args.file_out_ohe
    file_out_predict = args.file_out_predict
    target = args.target
    training_test_split_percent = args.training_test_split_percent
    file_in_config = args.file_in_config
    init_ohe_config(file_in_config)

    algorithms = set( get_algorithm_from_string(p.strip().upper()) for p in args.predictor)


    ohe_builder = OneHotEncoderBuilder(file_in_name)
    for ignore in args.ignore:
        ohe_builder.ignore(ignore)
    ohe = ohe_builder.build()
    data_frame, feature_name_list = ohe.one_hot_encode()
    ohe.write_ohe_csv(file_out_ohe)

    ohp_builder = OneHotPredictorBuilder(target, training_test_split_percent, data_frame)
    # Drops target and ignored from features
    features = ( f for f in feature_name_list if f not in args.ignore and f != target )
    for f in features:
        ohp_builder.add_feature(f)


    runner = Runner(ohp_builder, algorithms)
    runner.run_and_build_predictions()
    runner.write_predict_csv(file_out_predict)

if __name__ == '__main__':
    main()


