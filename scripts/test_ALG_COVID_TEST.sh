#!/usr/bin/env bash
set -e
    rm -f "csvs/test_data_all_algs_100_FULL_PREDICT.csv"
    touch "csvs/test_data_all_algs_100_FULL_PREDICT.csv"
    echo "alg,4_27_20,4_28_20,4_29_20,4_30_20,5_1_20,5_2_20,5_3_20,5_4_20,5_5_20,5_6_20, 5_7_20,5_8_20,classification_accuracy,f1_score,target" > "csvs/test_data_all_algs_100_FULL_PREDICT.csv"
for alg in    "MLPCLASSALPHA" "GNBAYES" "GNBAYESSMOOTHING"  ; do
    rm -rf "csvs/test_data_100_$alg/"
    mkdir -p "csvs/test_data_100_$alg"
    rm -f "csvs/test_data_100_$alg/test_data_100_FULL_PREDICT.csv"
    touch "csvs/test_data_100_$alg/test_data_100_FULL_PREDICT.csv"
    echo "4_27_20,4_28_20,4_29_20,4_30_20,5_1_20,5_2_20,5_3_20,5_4_20,5_5_20,5_6_20, 5_7_20,5_8_20,classification_accuracy,f1_score,target" > "csvs/test_data_100_$alg/test_data_100_FULL_PREDICT.csv"
    for I in 200 500 2000 4000 ; do
        for J in 200 500  2000 4000 ; do
        echo "Running $I $J"
       python -m discrete  \
             --file_in time_series_covid19_deaths_US_V9.csv \
              --dicretize uniform "$I" 4_27_20 \
              --dicretize uniform "$I" 4_28_20 \
              --dicretize uniform "$I" 4_29_20 \
              --dicretize uniform "$I" 4_30_20 \
              --dicretize uniform "$I" 5_1_20 \
              --dicretize uniform "$I" 5_2_20 \
              --dicretize uniform "$I" 5_3_20 \
              --dicretize uniform "$I" 5_4_20 \
              --dicretize uniform "$I" 5_5_20 \
              --dicretize uniform "$I" 5_6_20 \
              --dicretize uniform "$I" 5_7_20 \
              --dicretize uniform "$J" 5_8_20 \
              --file_out_discrete "csvs/test_data_100_$alg/test_data_100_DI_$I$J.csv" \
              --file_out "csvs/test_data_100_$alg/test_data_100_D_$I$J.csv"
       python -m predict  \
              --file_in "csvs/test_data_100_$alg/test_data_100_D_$I$J.csv" \
              --strategy none \
              --file_in_config config/ohe_config.yaml \
              --target 5_8_20_DISCRETE \
              --training_test_split_percent 70 \
              --predictor "$alg" \
              --score f1_score \
              --score classification_accuracy \
              --file_out "csvs/test_data_100_$alg/test_data_100_PREDICT_$I$J.csv"
       echo "$I,$I,$I,$I,$I,$I,$I,$I,$I,$I,$I,$J,$(tail -n +2 csvs/test_data_100_$alg/test_data_100_PREDICT_$I$J.csv)" >> "csvs/test_data_100_$alg/test_data_100_FULL_PREDICT.csv"
       echo "$alg,$I,$I,$I,$I,$I,$I,$I,$I,$I,$I,$I,$J,$(tail -n +2 csvs/test_data_100_$alg/test_data_100_PREDICT_$I$J.csv)" >> "csvs/test_data_all_algs_100_FULL_PREDICT.csv"

        done
    done
done