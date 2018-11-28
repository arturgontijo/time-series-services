from __future__ import print_function

import cntk as C

import datetime
import numpy as np
import os
import pandas as pd
# pip install git+https://github.com/pydata/pandas-datareader.git
from pandas_datareader import data
import pickle as pkl
import time
import traceback

C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components
pd.options.mode.chained_assignment = None  # default='warn'
# Set a random seed
np.random.seed(123)

data_file = os.path.join("data", "Stock", "stock_SPY.pkl")


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


def download(file_name):
    try:
        ts_data = get_stock_data("SPY", 2010, 1, 2, 2017, 1, 1)
    except:
        raise Exception("Data could not be downloaded")

    dir_name = os.path.dirname(file_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.isfile(file_name):
        print("Saving", file_name)
        with open(file_name, 'wb') as f:
            pkl.dump(ts_data, f, protocol=2)
    return ts_data


def create_model(net_input, num_output_classes, num_hidden_layers, hidden_layers_dim):
    h = net_input
    with C.layers.default_options(init=C.glorot_uniform()):
        for i in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim,
                               activation=C.relu)(h)
        r = C.layers.Dense(num_output_classes, activation=None)(h)
    return r


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"
    if mb % frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error * 100))
    return mb, training_loss, eval_error


def create_drawdowns(equity_curve):
    """
    Calculate the largest peak-to-trough drawdown of the PnL curve
    as well as the duration of the drawdown. Requires that the
    pnl_returns is a pandas Series.

    Parameters:
    pnl - A pandas Series representing period percentage returns.

    Returns:
    drawdown, duration - Highest peak-to-trough drawdown and duration.
    """

    # Calculate the cumulative returns curve
    # and set up the High Water Mark
    # Then create the drawdown and duration series
    hwm = [0]
    eq_idx = equity_curve.index
    drawdown = pd.Series(index = eq_idx)
    duration = pd.Series(index = eq_idx)

    # Loop over the index range
    for t in range(1, len(eq_idx)):
        cur_hwm = max(hwm[t-1], equity_curve[t])
        hwm.append(cur_hwm)
        drawdown[t]= (hwm[t] - equity_curve[t])
        duration[t]= 0 if drawdown[t] == 0 else duration[t-1] + 1
    return drawdown.max(), duration.max()


def main():

    try:
        # Check for data in local cache
        if os.path.exists(data_file):
            print("File already exists", data_file)
            ts_data = pd.read_pickle(data_file)
        else:
            # If not there we might be running in CNTK's test infrastructure
            if is_test():
                test_file = os.path.join(os.environ[envvar], 'Tutorials', 'data', 'stock', 'stock_SPY.pkl')
                if os.path.isfile(test_file):
                    print("Reading data from test data directory")
                    ts_data = pd.read_pickle(test_file)
                else:
                    print("Test data directory missing file", test_file)
                    print("Downloading data from Google Finance")
                    ts_data = download(data_file)
            else:
                # Local cache is not present and not test env
                # download the data from Google finance and cache it in a local directory
                # Please check if there is trade data for the chosen stock symbol during this period
                ts_data = download(data_file)

        # Feature name list
        predictor_names = []

        # Compute price difference as a feature
        ts_data["diff"] = np.abs((ts_data["Close"] - ts_data["Close"].shift(1)) / ts_data["Close"]).fillna(0)
        predictor_names.append("diff")

        # Compute the volume difference as a feature
        ts_data["v_diff"] = np.abs((ts_data["Volume"] - ts_data["Volume"].shift(1)) / ts_data["Volume"]).fillna(0)
        predictor_names.append("v_diff")

        # Compute the stock being up (1) or down (0) over different day offsets compared to current dat closing price
        num_days_back = 8

        for i in range(1, num_days_back + 1):
            data["p_" + str(i)] = np.where(ts_data["Close"] > ts_data["Close"].shift(i), 1, 0)  # i: number of look back days
            predictor_names.append("p_" + str(i))

        # If you want to save the file to your local drive
        # data.to_csv("PATH_TO_SAVE.csv")
        ts_data.head(10)

        ts_data["next_day"] = np.where(ts_data["Close"].shift(-1) > ts_data["Close"], 1, 0)
        ts_data["next_day_opposite"] = np.where(ts_data["next_day"] == 1, 0, 1)  # The label must be one-hot encoded

        # Establish the start and end date of our training timeseries (picked 2000 days before the market crash)
        training_data = ts_data["2001-02-05":"2009-01-20"]

        # We define our test data as: data["2008-01-02":]
        # This example allows to include data up to current date

        test_data = ts_data["2009-01-20":"2016-12-29"]
        training_features = np.asarray(training_data[predictor_names], dtype="float32")
        training_labels = np.asarray(training_data[["next_day", "next_day_opposite"]], dtype="float32")

        # Lets build the network
        input_dim = 2 + num_days_back
        num_output_classes = 2  # Remember we need to have 2 since we are trying to classify if the market goes up or down 1 hot encoded
        num_hidden_layers = 2
        hidden_layers_dim = 2 + num_days_back
        input_dynamic_axes = [C.Axis.default_batch_axis()]
        net_input = C.input_variable(input_dim, dynamic_axes=input_dynamic_axes)
        label = C.input_variable(num_output_classes, dynamic_axes=input_dynamic_axes)

        z = create_model(net_input, num_output_classes, num_hidden_layers, hidden_layers_dim)
        loss = C.cross_entropy_with_softmax(z, label)
        label_error = C.classification_error(z, label)
        lr_per_minibatch = C.learning_parameter_schedule(0.125)
        trainer = C.Trainer(z, (loss, label_error), [C.sgd(z.parameters, lr=lr_per_minibatch)])

        # Initialize the parameters for the trainer, we will train in large minibatches in sequential order
        minibatch_size = 100
        num_minibatches = len(training_data.index) // minibatch_size

        # Run the trainer on and perform model training
        training_progress_output_freq = 1

        # Visualize the loss over minibatch
        plotdata = {"batchsize": [], "loss": [], "error": []}

        # It is key that we make only one pass through the data linearly in time
        num_passes = 1

        # Train our neural network
        tf = np.split(training_features, num_minibatches)
        tl = np.split(training_labels, num_minibatches)

        print("Number of mini batches")
        print(len(tf))
        print("The shape of the training feature minibatch")
        print(tf[0].shape)

        for i in range(num_minibatches * num_passes):  # multiply by the
            features = np.ascontiguousarray(tf[i % num_minibatches])
            labels = np.ascontiguousarray(tl[i % num_minibatches])

            # Specify the mapping of input variables in the model to actual minibatch data to be trained with
            trainer.train_minibatch({net_input: features, label: labels})
            batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
            if not (loss == "NA" or error == "NA"):
                plotdata["batchsize"].append(batchsize)
                plotdata["loss"].append(loss)
                plotdata["error"].append(error)

        # Now that we have trained the net, and we will do out of sample test to see how we did.
        # and then more importantly analyze how that set did

        test_features = np.ascontiguousarray(test_data[predictor_names], dtype="float32")
        test_labels = np.ascontiguousarray(test_data[["next_day", "next_day_opposite"]], dtype="float32")

        avg_error = trainer.test_minibatch({net_input: test_features, label: test_labels})
        print("Average error: {0:2.2f}%".format(avg_error * 100))

        out = C.softmax(z)
        predicted_label_prob = out.eval({net_input: test_features})
        test_data["p_up"] = pd.Series(predicted_label_prob[:, 0], index=test_data.index)
        test_data["p_down"] = predicted_label_prob[:, 1]
        test_data['long_entries'] = np.where((test_data.p_up > 0.55), 1, 0)
        test_data['short_entries'] = np.where((test_data.p_down > 0.55), -1, 0)
        test_data['positions'] = test_data['long_entries'].fillna(0) + test_data['short_entries'].fillna(0)

        test_data["pnl"] = test_data["Close"].diff().shift(-1).fillna(0) * test_data["positions"] / np.where(
            test_data["Close"] != 0, test_data["Close"], 1)
        test_data["perc"] = (test_data["Close"] - test_data["Close"].shift(1)) / test_data["Close"].shift(1)
        monthly = test_data.pnl.resample("M").sum()
        monthly_spy = test_data["perc"].resample("M").sum()
        avg_return = np.mean(monthly)
        std_return = np.std(monthly)
        sharpe = np.sqrt(12) * avg_return / std_return
        drawdown = create_drawdowns(monthly.cumsum())
        spy_drawdown = create_drawdowns(monthly_spy.cumsum())
        print("TRADING STATS")
        print("AVG Monthly Return :: " + "{0:.2f}".format(round(avg_return * 100, 2)) + "%")
        print("STD Monthly        :: " + "{0:.2f}".format(round(std_return * 100, 2)) + "%")
        print("SHARPE             :: " + "{0:.2f}".format(round(sharpe, 2)))
        print(
            "MAX DRAWDOWN       :: " + "{0:.2f}".format(round(drawdown[0] * 100, 2)) + "%, " + str(drawdown[1]) + " months")
        print("Correlation to SPY :: " + "{0:.2f}".format(round(np.corrcoef(test_data["pnl"], test_data["diff"])[0][1], 2)))
        print("NUMBER OF TRADES   :: " + str(np.sum(test_data.positions.abs())))
        print("TOTAL TRADING DAYS :: " + str(len(data)))
        print("SPY MONTHLY RETURN :: " + "{0:.2f}".format(round(monthly_spy.mean() * 100, 2)) + "%")
        print("SPY STD RETURN     :: " + "{0:.2f}".format(round(monthly_spy.std() * 100, 2)) + "%")
        print("SPY SHARPE         :: " + "{0:.2f}".format(round(monthly_spy.mean() / monthly_spy.std() * np.sqrt(12), 2)))
        print("SPY DRAWDOWN       :: " + "{0:.2f}".format(round(spy_drawdown[0] * 100, 2)) + "%, " + str(
            spy_drawdown[1]) + " months")

        print(drawdown[0])

        # ===============================================

        test_data["p_up"] = pd.Series(predicted_label_prob[:, 0], index=test_data.index)
        test_data["p_down"] = predicted_label_prob[:, 1]
        test_data['long_entries'] = np.where((test_data.p_up > 0.50), 1, 0)
        test_data['short_entries'] = np.where((test_data.p_down > 0.50), -1, 0)
        test_data['positions'] = test_data['long_entries'].fillna(0) + test_data['short_entries'].fillna(0)
        test_data["pnl"] = test_data["Close"].diff().shift(-1).fillna(0) * test_data["positions"] / np.where(
            test_data["Close"] != 0, test_data["Close"], 1)
        monthly = test_data.pnl.resample("M").sum()
        avg_return = np.mean(monthly)
        std_return = np.std(monthly)
        sharpe = np.sqrt(12) * avg_return / std_return
        drawdown = create_drawdowns(monthly.cumsum())

        print("TRADING STATS")
        print("AVG Monthly Return :: " + "{0:.2f}".format(round(avg_return * 100, 2)) + "%")
        print("STD Monthly        :: " + "{0:.2f}".format(round(std_return * 100, 2)) + "%")
        print("SHARPE             :: " + "{0:.2f}".format(round(sharpe, 2)))
        print(
            "MAX DRAWDOWN       :: " + "{0:.2f}".format(round(drawdown[0] * 100, 2)) + "%, " + str(drawdown[1]) + " months")
        print("Correlation to SPY :: " + "{0:.2f}".format(round(np.corrcoef(test_data["pnl"], test_data["diff"])[0][1], 2)))

    except:
        traceback.print_exc()


if __name__ == "__main__":
    main()




















