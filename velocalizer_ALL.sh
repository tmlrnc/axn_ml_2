#!/usr/bin/env bash

python -m velocalizer  \
  --file_in csvs/12AxilientAdmin_V1.csv \
  --file_out csvs/12AxilientAdmin_V1_SAMPLE.csv \
  --noise_threshold 300 \
  --ignore Person_Client_ID \
  --ignore Person_First_Nam \
  --ignore Person_Last_Nam \
  --ignore Person_Address_1 \
  --ignore Person_City \
  --features All



