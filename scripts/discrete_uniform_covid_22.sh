#!/usr/bin/env bash
set -e
rm -f csvs/data/18_FULL_PREDICT.csv
touch csvs/data/18_FULL_PREDICT.csv
echo "POLLUTION_INDEX_BINS,MORT_RATION_BINS,f1_score,classification_accuracy,recall,target" > csvs/data/18_FULL_PREDICT.csv
for I in 3 4 5 6 7 ; do
    for J in 3 4 5 6 7 ; do
    echo "Running $I $J"
   python -m discrete  \
         --file_in csvs/data/18_POLL_MORT.csv \
          --dicretize uniform "$I" POLLUTION_INDEX \
          --dicretize uniform "$J" MORT_RATIO \
          --file_out_discrete "csvs/data/18_POLL_MORT_D_$I$J.csv" \
          --file_out "csvs/data/18_POLL_MORT_$I$J.csv"
   python -m predict  \
          --file_in "csvs/data/18_POLL_MORT_$I$J.csv" \
          --strategy none \
          --file_in_config config/ohe_config.yaml \
          --target MORT_RATIO_DISCRETE \
          --training_test_split_percent 70 \
          --predictor SVM \
          --score f1_score \
          --score classification_accuracy \
          --score recall \
          --file_out "csvs/data/18_POLL_MORT_PREDICT_$I$J.csv"
   echo "$I,$J,$(tail -n +2 csvs/data/18_POLL_MORT_PREDICT_$I$J.csv)" >> csvs/data/18_FULL_PREDICT.csv
    done
done