import argparse

from ohe.config import init_ohe_config
from ohe.discretize import K_Means, normalizer
from sklearn.preprocessing import KBinsDiscretizer

from ohe.binize import VL_Binizer

import pandas as pd
import numpy

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
    args = parse_command_line()
    file_in_name = args.file_in
    file_out_ohe = args.file_out_ohe
    file_out_predict = args.file_out_predict
    training_test_split_percent = args.training_test_split_percent
    file_in_config = args.file_in_config
    ohe_only = args.ohe_only



    ######################################################################


    df = pd.read_csv(file_in_name, sep=',', header=None)
    X = df.to_numpy()


    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    est.fit(X)
    Xt = est.transform(X)


    print("Xt " + str(Xt))
    print("bin_edges_ " + str(est.bin_edges_))


    bin = VL_Binizer(n_bins=5, encode='ordinal', strategy='uniform')
    bin.fit(X)
    Xt_VL = bin.transform(X)

    print("Xt_VL " + str(Xt_VL))
    print("bin_edges_ " + str(bin.bin_edges_))

    print('*************************************')

    bin_AS = VL_Binizer(n_bins=5, encode='ordinal', strategy='analyst_supervised', edge_array=[1,5.,10.])
    bin_AS.fit(X)
    Xt_VL = bin_AS.transform(X)

    print("analyst_supervised Xt_VL " + str(Xt_VL))
    print("analyst_supervised bin_edges_ " + str(bin_AS.bin_edges_))

    print('*************************************')

    bin_AS = VL_Binizer(n_bins=5, encode='ordinal', strategy='analyst_supervised', edge_array=[3.,9.])
    bin_AS.fit(X)
    Xt_VL = bin_AS.transform(X)

    print("analyst_supervised Xt_VL " + str(Xt_VL))
    print("analyst_supervised bin_edges_ " + str(bin_AS.bin_edges_))


    print('*************************************')

    bin_AS = VL_Binizer(n_bins=5, encode='ordinal', strategy='analyst_supervised', edge_array=[55.])
    bin_AS.fit(X)
    Xt_VL = bin_AS.transform(X)

    print("analyst_supervised Xt_VL " + str(Xt_VL))
    print("analyst_supervised bin_edges_ " + str(bin_AS.bin_edges_))


    exit()


    normalizer.fit(X)
    # apply transform
    normalized = normalizer.transform(X)
    # inverse transform
    inverse = normalizer.inverse_transform(normalized)

    print("normalized " + str(normalized))
    print("inverse " + str(inverse))


    model = K_Means(k=4,tol=0.001, max_iter=200)
    model.fit(X)

    for centroid in model.centroids:
        print('*************************************')
        print(model.centroids[centroid][0])

    mode_norm = K_Means(k=4, tol=0.001, max_iter=200)
    mode_norm.fit(normalized)

    my_cent = []
    for centroid in mode_norm.centroids:
        print('************************************* NORMALIZED')
        print(mode_norm.centroids[centroid][0])
        my_cent.append(mode_norm.centroids[centroid][0])

    arr = numpy.array(my_cent)
    myarr = arr.reshape(-1,1)
    my_cent_i_x = normalizer.inverse_transform(myarr)
    for x in my_cent_i_x:
        print(x)


    df = pd.read_csv(file_in_name, sep=',', header=None)
    Y = df.to_numpy()





    init_ohe_config(file_in_config)
    algorithms = set( get_algorithm_from_string(p.strip().upper()) for p in args.predictor)
    ohe_builder = OneHotEncoderBuilder(file_in_name)
    score_list = args.score
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
            runner.run_and_build_predictions(score_list)
            runner.write_predict_csv(file_out_predict, new_target,write_header_flag)
            write_header_flag = 0




if __name__ == '__main__':
    main()


