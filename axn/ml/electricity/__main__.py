"""
main for discrete
"""
# pylint: disable=unused-variable
# pylint: disable=line-too-long
# pylint: disable=duplicate-code

import argparse

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


#https://github.com/rvgramillano/springboard_portfolio/blob/master/Electricity_Demand/modeling/modeling.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor


def evaluate(model, X_test, y_test, X_train, y_train, m_name):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Compute and print the metrics
    r2_test = model.score(X_test, y_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    r2_train = model.score(X_train, y_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

    print
    m_name

    print
    '---------------------'
    print
    'Train R^2: %.4f' % r2_train
    print
    'Train Root MSE: %.4f' % rmse_train

    print
    '---------------------'
    print
    'Test R^2: %.4f' % r2_test
    print
    'Test Root MSE: %.4f' % rmse_test

    return r2_test, rmse_test



def series_to_supervised(data,  col_names, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
     data: Sequence of observations as a list or NumPy array.
     n_in: Number of lag observations as input (X).
     n_out: Number of observations as output (y).
     dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
     Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (col_names[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    if i == 0:
        names += [('%s(t)' % (col_names[j])) for j in range(n_vars)]
    else:
        names += [('%s(t+%d)' % (col_names[j], i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor





import pandas as pd
import statsmodels.api as sm
import pickle
import numpy as np
import scipy.stats as stats

def multiple_regression(df, name):
    X = df[[col for col in df.columns if col != 'demand']]
    y = df['demand']
    X = sm.add_constant(X)  ## let's add an intercept (beta_0) to our model

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    print('------------------%s------------------' % name)
    print(model.summary())
    return model


import csv
from six.moves import cPickle as pickle
import numpy as np


def pkl_to_csv(path_pickle,path_csv):

    x = []

    pickle.loads(path_pickle)
    exit()

    pd.read_pickle(path_pickle)


    print(pd)
    df = pd.DataFrame()

    df.to_csv(path_csv)
    with open(path_pickle,'rb') as f:
        x = pickle.load(f)

    with open(path_csv,'w') as f:
        writer = csv.writer(f)
        for line in x: writer.writerow(line)

def parse_command_line():
    """
    reads the command line args
    """
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_in',
        help='raw csv file input to be predicted. Must be a csv file where first row has column header '
             'names. Must include time series date columns - like MM/DD/YY (7/3/20) ')
    parser.add_argument(
        '--file_out',
        help='csv file output encoded using one-hot one-of-K encoding scheme')
    args = parser.parse_args()
    return args



def main():
    """
      """
    # pylint: disable=duplicate-code
    WORKING_DIR = '/Users/tomlorenc/Downloads/'
    file_in = '/Users/tomlorenc/Downloads/LA_df.pkl'

    la_df = pd.read_pickle(file_in)
    print(la_df)

    X = la_df[[col for col in la_df.columns if col != 'demand']]
    y = la_df['demand']
    X = sm.add_constant(X)  ## let's add an intercept (beta_0) to our model

    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    print("*******************")
    print(model.summary())

    exit()
    file_out = '/Users/tomlorenc/Downloads/LA.csv'
    #pkl_to_csv(file_in,file_out)

    m_la = multiple_regression(la_df, 'LOS ANGELES')

    # drop high p-value columns and save
    la_df = la_df.drop(['hourlywindspeed', 'hourlyheatingdegrees', 'hourlyskyconditions_BKN', 'hourlyskyconditions_FEW',
                        'hourlyskyconditions_OVC', 'hourlyskyconditions_SCT'], axis=1)



    #la_df.to_pickle(WORKING_DIR + 'LA_df_final_v2.pkl')

    print("MBA --- END ")

if __name__ == '__main__':
    main()
