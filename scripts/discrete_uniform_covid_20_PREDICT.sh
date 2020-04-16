#!/usr/bin/env bash
for I in 3 4 5; do
    for J in 3 4 5; do
    echo "Running $I $J"
	python -m discrete  \
	      --file_in csvs/18_POLL_MORT.csv \
	       --dicretize uniform "$I" POLLUTION_INDEX \
	       --dicretize uniform "$J" MORT_RATIO \
	       --file_out_discrete "csvs/18_POLL_MORT_D_$I$J.csv" \
	       --file_out "csvs/18_POLL_MORT_$I$J.csv"
	python -m predict  \
	       --file_in "csvs/18_POLL_MORT_$I$J.csv" \
	       --strategy none \
	       --file_in_config config/ohe_config.yaml \
	       --target MORT_RATIO_DISCRETE \
	       --training_test_split_percent 70 \
	       --predictor SVM \
	       --score f1_score \
	       --score classification_accuracy \
	       --score recall \
	       --file_out "csvs/18_POLL_MORT_PREDICT_$I$J.csv"
    done

done