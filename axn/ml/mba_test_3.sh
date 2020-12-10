#!/usr/bin/env bash


python -m zeroblank  \
  --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3.csv  \
  --file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero.csv


python -m ohe  \
  --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero.csv  \
  --ignore ID \
  --file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero_ohe.csv


python -m cut_id  \
  --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero_ohe.csv  \
  --file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero_ohe_cut.csv

python -m market_basket_analysis  \
  --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero_ohe_cut.csv  \
  --file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero_ohe_cut_mba.csv

python -m cut_first  \
  --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero_ohe_cut_mba.csv  \
  --file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_OUT.csv

