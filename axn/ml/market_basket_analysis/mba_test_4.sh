#!/usr/bin/env bash




python -m market_basket_analysis  \
  --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero_ohe_cut.csv  \
  --file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero_ohe_cut_mba.csv

python -m cut_first  \
  --file_in /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_zero_ohe_cut_mba.csv  \
  --file_out /Users/tomlorenc/Sites/VL_standard/ml/axn/ml/market_basket_analysis/test_data/MBA_IN_TEST3_OUT.csv

