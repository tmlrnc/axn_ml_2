#!/usr/bin/env bash

python -m ohe  \
  --file_in csvs/12AxilientAdmin.csv \
  --file_out_ohe csvs/12AxilientAdmin_OHE.csv  \
  --file_out_predict csvs/12_V1_PRED.csv \
  --file_in_config config/ohe_config_RUN1.yaml \
  --ignore Person_Client_ID \
  --ignore Person_First_Nam \
  --ignore Person_Last_Nam \
  --ignore Person_Address_1 \
  --ignore Person_City \
  --ignore Person_Zip \
  --target axn_Ppl_MarriedPrtner_cod \
  --training_test_split_percent 70 \
  --ohe_only NO \
  --predictor SVM

