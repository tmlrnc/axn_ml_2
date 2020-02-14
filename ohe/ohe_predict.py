import argparse

from ohe.config import init_ohe_config
from ohe.predictor import OneHotPredictorBuilder
from ohe.encoder import OneHotEncoderBuilder

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in')
    parser.add_argument('--file_out_ohe')
    parser.add_argument('--file_out_predict')
    parser.add_argument('--target')
    parser.add_argument('--training_test_split_percent')
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

    ohe_builder = OneHotEncoderBuilder(file_in_name)
    for ignore in args.ignore:
        ohe_builder.ignore(ignore)
    ohe = ohe_builder.build()
    one_hot_encode_object, feature_name_list = ohe.one_hot_encode()
    ohe.write_ohe_csv(file_out_ohe)

    ohp_builder = OneHotPredictorBuilder(target,training_test_split_percent)
    for predictor in args.predictor:
        ohp_builder.add_predictor(predictor)
    ohp = ohp_builder.build()
    ohp.predict(one_hot_encode_object,feature_name_list, target)
    ohp.write_predict_csv(file_out_predict)

if __name__ == '__main__':
    main()


