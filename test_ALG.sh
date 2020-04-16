#!/usr/bin/env bash
set -e
    rm -f "csvs/test_data_all_algs_100_FULL_PREDICT.csv"
    touch "csvs/test_data_all_algs_100_FULL_PREDICT.csv"
    echo "alg,A,B,classification_accuracy,f1_score,target" > "csvs/test_data_all_algs_100_FULL_PREDICT.csv"
for alg in "RFR" "LR" "MLP" "SVM" "LLM" "NUSVM" "BFRA" "NUSVMSIG" "LSQLDA" "MULTICLASSLR" "RIDGEREGRESSION" "LASSOMODEL" "BAYESIANRIDGE"; do
    rm -rf "csvs/test_data_100_$alg/"
    mkdir -p "csvs/test_data_100_$alg"
    rm -f "csvs/test_data_100_$alg/test_data_100_FULL_PREDICT.csv"
    touch "csvs/test_data_100_$alg/test_data_100_FULL_PREDICT.csv"
    echo "A,B,classification_accuracy,f1_score,target" > "csvs/test_data_100_$alg/test_data_100_FULL_PREDICT.csv"
    for I in 2 3 4 5 ; do
        for J in 2 3 4 5 ; do
        echo "Running $I $J"
       python -m discrete  \
             --file_in csvs/test_data_100.csv \
              --dicretize uniform "$I" A \
              --dicretize uniform "$J" B \
              --file_out_discrete "csvs/test_data_100_$alg/test_data_100_DI_$I$J.csv" \
              --file_out "csvs/test_data_100_$alg/test_data_100_D_$I$J.csv"
       python -m predict  \
              --file_in "csvs/test_data_100_$alg/test_data_100_D_$I$J.csv" \
              --strategy none \
              --file_in_config config/ohe_config.yaml \
              --target B_DISCRETE \
              --training_test_split_percent 70 \
              --predictor "$alg" \
              --score f1_score \
              --score classification_accuracy \
              --file_out "csvs/test_data_100_$alg/test_data_100_PREDICT_$I$J.csv"
       echo "$I,$J,$(tail -n +2 csvs/test_data_100_$alg/test_data_100_PREDICT_$I$J.csv)" >> "csvs/test_data_100_$alg/test_data_100_FULL_PREDICT.csv"
       echo "$alg, $I,$J,$(tail -n +2 csvs/test_data_100_$alg/test_data_100_PREDICT_$I$J.csv)" >> "csvs/test_data_all_algs_100_FULL_PREDICT.csv"

        done
    done
done