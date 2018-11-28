from __future__ import print_function

import cntk as C

import datetime
import numpy as np
import os
import pandas as pd
from pandas_datareader import data
import pickle as pkl
import time

C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components
pd.options.mode.chained_assignment = None  # default='warn'
# Set a random seed
np.random.seed(123)


def get_stock_data(contract, s_year, s_month, s_day, e_year, e_month, e_day):
    """
    Args:
        contract (str): the name of the stock/etf
        s_year (int): start year for data
        s_month (int): start month
        s_day (int): start day
        e_year (int): end year
        e_month (int): end month
        e_day (int): end day
    Returns:
        Pandas Dataframe: Daily OHLCV bars
    """
    start = datetime.datetime(s_year, s_month, s_day)
    end = datetime.datetime(e_year, e_month, e_day)

    retry_cnt, max_num_retry = 0, 3

    while (retry_cnt < max_num_retry):
        try:
            bars = data.DataReader(contract, "iex", start, end)
            return bars
        except:
            retry_cnt += 1
            time.sleep(np.random.randint(1, 10))

    print("iex Finance is not reachable")
    raise Exception('iex Finance is not reachable')


# We search in cached stock data set with symbol SPY.
# Check for an environment variable defined in CNTK's test infrastructure
envvar = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY'


def is_test(): return envvar in os.environ


def download(data_file):
    try:
        data = get_stock_data("SPY", 2015, 1, 2, 2017, 1, 1)
    except:
        raise Exception("Data could not be downloaded")

    dir = os.path.dirname(data_file)

    if not os.path.exists(dir):
        os.makedirs(dir)

    if not os.path.isfile(data_file):
        print("Saving", data_file)
        with open(data_file, 'wb') as f:
            pkl.dump(data, f, protocol=2)
    return data


data_file = os.path.join("data", "Stock", "stock_SPY.pkl")

# Check for data in local cache
if os.path.exists(data_file):
    print("File already exists", data_file)
    data = pd.read_pickle(data_file)
else:
    # If not there we might be running in CNTK's test infrastructure
    if is_test():
        test_file = os.path.join(os.environ[envvar], 'Tutorials', 'data', 'stock', 'stock_SPY.pkl')
        if os.path.isfile(test_file):
            print("Reading data from test data directory")
            data = pd.read_pickle(test_file)
        else:
            print("Test data directory missing file", test_file)
            print("Downloading data from Google Finance")
            data = download(data_file)
    else:
        # Local cache is not present and not test env
        # download the data from Google finance and cache it in a local directory
        # Please check if there is trade data for the chosen stock symbol during this period
        data = download(data_file)

# Feature name list
predictor_names = []

# Compute price difference as a feature
data["diff"] = np.abs((data["Close"] - data["Close"].shift(1)) / data["Close"]).fillna(0)
predictor_names.append("diff")

# Compute the volume difference as a feature
data["v_diff"] = np.abs((data["Volume"] - data["Volume"].shift(1)) / data["Volume"]).fillna(0)
predictor_names.append("v_diff")

# Compute the stock being up (1) or down (0) over different day offsets compared to current dat closing price
num_days_back = 8

for i in range(1, num_days_back + 1):
    data["p_" + str(i)] = np.where(data["Close"] > data["Close"].shift(i), 1, 0)  # i: number of look back days
    predictor_names.append("p_" + str(i))

# If you want to save the file to your local drive
# data.to_csv("PATH_TO_SAVE.csv")
data.head(10)




















