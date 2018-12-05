import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
import time

import cntk as C

from pandas_datareader import data as pd_data
import datetime

from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from saxpy.znorm import znorm

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)

# to make things reproduceable, seed random
np.random.seed(0)

isFast = True

# we need around 2000 epochs to see good accuracy. For testing 100 epochs will do.
EPOCHS = 100 if isFast else 2000


def generate_my_data(data_file, time_steps, time_shift):
    data = pd.read_csv(data_file, dtype=np.float32)

    rnn_x = []
    for i in range(len(data) - time_steps + 1):
        rnn_x.append(data["x"].iloc[i: i + time_steps].as_matrix())
    rnn_x = np.array(rnn_x)

    # Reshape or rearrange the data from row to columns
    # to be compatible with the input needed by the LSTM model
    # which expects 1 float per time point in a given batch
    rnn_x = rnn_x.reshape(rnn_x.shape + (1,))

    rnn_y = data["y"].values
    rnn_y = rnn_y[time_steps - 1:]

    # Reshape or rearrange the data from row to columns
    # to match the input shape
    rnn_y = rnn_y.reshape(rnn_y.shape + (1,))

    return split_data(rnn_x), split_data(rnn_y)


def get_my_data(n, m):
    N = n  # input: N subsequent values
    M = m  # output: predict 1 value M steps ahead
    return generate_my_data(input("CSV path: "), N, M)


# =============================================================================================


def generate_solar_data(input_url, time_steps, normalize=1, val_size=0.1, test_size=0.1):
    """
    generate sequences to feed to rnn based on data frame with solar panel data
    the csv has the format: time ,solar.current, solar.total
     (solar.current is the current output in Watt, solar.total is the total production
      for the day so far in Watt hours)
    """
    # try to find the data file local. If it doesn"t exists download it.
    if "http://" in input_url or "https://" in input_url:
        cache_path = os.path.join("data", "iot")
        cache_file = os.path.join(cache_path, "solar.csv")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if not os.path.exists(cache_file):
            urlretrieve(input_url, cache_file)
            print("downloaded data successfully from ", input_url)
        else:
            print("using cache for ", input_url)
    else:
        cache_file = input_url

    df = pd.read_csv(cache_file, dtype=np.float32)

    # split the dataset into train, validation and test sets on day boundaries
    val_size = int(len(df) * val_size)
    test_size = int(len(df) * test_size)
    next_val = 0
    next_test = 0

    result_x = {"train": [], "val": [], "test": []}
    result_y = {"train": [], "val": [], "test": []}

    # generate sequences a day at a time
    for i, total in enumerate(df["x"].values):
        if i >= next_val:
            current_set = "val"
            next_val = i + int(len(df) / val_size)
        elif i >= next_test:
            current_set = "test"
            next_test = i + int(len(df) / test_size)
        else:
            current_set = "train"
        for j in range(2, time_steps):
            result_x[current_set].append(df["x"].values[0:j])
            result_y[current_set].append([np.array(df["y"].values[i])])
            if j >= time_steps:
                break

    # make result_y a numpy array
    for ds in ["train", "val", "test"]:
        result_y[ds] = np.array(result_y[ds])
    return result_x, result_y


def generate_solar_data_old(input_url, time_steps, normalize=1, val_size=0.1, test_size=0.1):
    """
    generate sequences to feed to rnn based on data frame with solar panel data
    the csv has the format: time ,solar.current, solar.total
     (solar.current is the current output in Watt, solar.total is the total production
      for the day so far in Watt hours)
    """
    # try to find the data file local. If it doesn"t exists download it.
    if "http://" in input_url or "https://" in input_url:
        cache_path = os.path.join("data", "iot")
        cache_file = os.path.join(cache_path, "solar.csv")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if not os.path.exists(cache_file):
            urlretrieve(input_url, cache_file)
            print("downloaded data successfully from ", input_url)
        else:
            print("using cache for ", input_url)
    else:
        cache_file = input_url

    df = pd.read_csv(cache_file, index_col="time", parse_dates=["time"], dtype=np.float32)

    df["date"] = df.index.date

    # normalize data
    df["solar.current"] /= normalize
    df["solar.total"] /= normalize

    # group by day, find the max for a day and add a new column .max
    grouped = df.groupby(df.index.date).max()
    grouped.columns = ["solar.current.max", "solar.total.max", "date"]

    # merge continuous readings and daily max values into a single frame
    df_merged = pd.merge(df, grouped, right_index=True, on="date")
    df_merged = df_merged[["solar.current", "solar.total",
                           "solar.current.max", "solar.total.max"]]
    # we group by day so we can process a day at a time.
    grouped = df_merged.groupby(df_merged.index.date)
    per_day = []
    for _, group in grouped:
        per_day.append(group)

    # split the dataset into train, validation and test sets on day boundaries
    val_size = int(len(per_day) * val_size)
    test_size = int(len(per_day) * test_size)
    next_val = 0
    next_test = 0

    result_x = {"train": [], "val": [], "test": []}
    result_y = {"train": [], "val": [], "test": []}

    # generate sequences a day at a time
    for i, day in enumerate(per_day):
        # if we have less than 8 datapoints for a day we skip over the
        # day assuming something is missing in the raw data
        total = day["solar.total"].values
        if len(total) < 8:
            continue
        if i >= next_val:
            current_set = "val"
            next_val = i + int(len(per_day) / val_size)
        elif i >= next_test:
            current_set = "test"
            next_test = i + int(len(per_day) / test_size)
        else:
            current_set = "train"
        max_total_for_day = np.array(day["solar.total.max"].values[0])
        for j in range(2, len(total)):
            result_x[current_set].append(total[0:j])
            result_y[current_set].append([max_total_for_day])
            if j >= time_steps:
                break
    # make result_y a numpy array
    for ds in ["train", "val", "test"]:
        result_y[ds] = np.array(result_y[ds])
    return result_x, result_y


def next_batch(x, y, ds, batch_size):
    """get the next batch for training"""

    def as_batch(data, start, count):
        return data[start:start + count]

    for i in range(0, len(x[ds]), batch_size):
        yield as_batch(x[ds], i, batch_size), as_batch(y[ds], i, batch_size)


def create_model(x, h_dims):
    """Create the model for time series prediction"""
    with C.layers.default_options(initial_state = 0.1):
        m = C.layers.Recurrence(C.layers.LSTM(h_dims))(x)
        m = C.sequence.last(m)
        m = C.layers.Dropout(0.2)(m)
        m = C.layers.Dense(1)(m)
        return m


# validate
def get_mse(trainer, x_label, x, y, batch_size, l_label, labeltxt):
    result = 0.0
    for x1, y1 in next_batch(x, y, labeltxt, batch_size):
        eval_error = trainer.test_minibatch({x_label: x1, l_label: y1})
        result += eval_error
    return result/len(x[labeltxt])


# ======================================== SIN ================================================
def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits np.array into training, validation and test
    """
    pos_test = int(len(data) * (1 - test_size))
    pos_val = int(len(data[:pos_test]) * (1 - val_size))

    train, val, test = data[:pos_val], data[pos_val:pos_test], data[pos_test:]

    return {"train": train, "val": val, "test": test}


def generate_data(fct, x, time_steps, time_shift):
    """
    generate sequences to feed to rnn for fct(x)
    """
    data = fct(x)
    print("data=fct(x): ", data)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(dict(a=data[0:len(data) - time_shift],
                                 b=data[time_shift:]))
    print("data(pandas): ", data)
    rnn_x = []
    for i in range(len(data) - time_steps + 1):
        rnn_x.append(data['a'].iloc[i: i + time_steps].as_matrix())
    rnn_x = np.array(rnn_x)

    # Reshape or rearrange the data from row to columns
    # to be compatible with the input needed by the LSTM model
    # which expects 1 float per time point in a given batch
    rnn_x = rnn_x.reshape(rnn_x.shape + (1,))

    rnn_y = data['b'].values
    rnn_y = rnn_y[time_steps - 1:]

    # Reshape or rearrange the data from row to columns
    # to match the input shape
    rnn_y = rnn_y.reshape(rnn_y.shape + (1,))

    return split_data(rnn_x), split_data(rnn_y)


def get_sin(n, m, total_len):
    N = n  # input: N subsequent values
    M = m  # output: predict 1 value M steps ahead
    return generate_data(np.sin, np.linspace(0, 100, total_len, dtype=np.float32), N, M)
# =============================================================================================


def get_solar_old(t, n):
    # "https://www.cntk.ai/jup/dat/solar.csv"
    return generate_solar_data_old(input("CSV path: "), t, normalize=n)
# =============================================================================================


def get_solar(t, n):
    return generate_solar_data(input("CSV file: "), t, normalize=n)


def main():

    sax_window_len = int(input("SAX Window Length: "))
    H_DIMS = sax_window_len

    sax_alphabet_len = int(input("SAX Alphabet Length: "))


    source = input("Source: ")
    if "yahoo" in source:
        ts_data = pd_data.DataReader("SPY", source, "2000-01-01", datetime.datetime.now())
    else:
        ts_data = pd.read_csv(source, index_col="date", parse_dates=["date"], dtype=np.float32)

    sax_seq = ts_to_string(znorm(ts_data), cuts_for_asize(sax_alphabet_len))

    # input sequences
    x = C.sequence.input_variable(1)

    model_file = "{}_epochs.model".format(EPOCHS)

    if not os.path.exists(model_file):
        print("Training model {}...".format(model_file))

        # create the model
        z = create_model(x, H_DIMS)

        # expected output (label), also the dynamic axes of the model output
        # is specified as the model of the label input
        var_l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")

        # the learning rate
        learning_rate = 0.005
        lr_schedule = C.learning_parameter_schedule(learning_rate)

        # loss function
        loss = C.squared_error(z, var_l)

        # use squared error to determine error for now
        error = C.squared_error(z, var_l)

        # use adam optimizer
        momentum_schedule = C.momentum_schedule(0.9, minibatch_size=BATCH_SIZE)
        learner = C.fsadagrad(z.parameters,
                              lr=lr_schedule,
                              momentum=momentum_schedule)
        trainer = C.Trainer(z, (loss, error), [learner])

        # training
        loss_summary = []

        start = time.time()
        for epoch in range(0, EPOCHS):
            for x_batch, l_batch in next_batch(X, Y, "train", BATCH_SIZE):
                trainer.train_minibatch({x: x_batch, var_l: l_batch})

            if epoch % (EPOCHS / 10) == 0:
                training_loss = trainer.previous_minibatch_loss_average
                loss_summary.append(training_loss)
                print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))

        print("Training took {:.1f} sec".format(time.time() - start))

        # Print the train, validation and test errors
        for labeltxt in ["train", "val", "test"]:
            print("mse for {}: {:.6f}".format(labeltxt, get_mse(trainer, x, X, Y, BATCH_SIZE, var_l, labeltxt)))

        z.save(model_file)

    else:
        z = C.load_model(model_file)
        x = cntk.logging.find_all_with_name(z, "")[-1]

    # Print out all layers in the model
    print("Loading {} and printing all nodes:".format(model_file))
    node_outputs = cntk.logging.find_all_with_name(z, "")
    for n in node_outputs:
        print("  {}".format(n))

    # predict
    # f, a = plt.subplots(2, 1, figsize=(12, 8))
    for j, ds in enumerate(["val", "test"]):
        fig = plt.figure()
        a = fig.add_subplot(2, 1, 1)
        results = []
        for x_batch, y_batch in next_batch(X, Y, ds, BATCH_SIZE_TEST):
            pred = z.eval({x: x_batch})
            results.extend(pred[:, 0])
        # because we normalized the input data we need to multiply the prediction
        # with SCALER to get the real values.
        a.plot((Y[ds] * NORMALIZE).flatten(), label=ds + " raw")
        a.plot(np.array(results) * NORMALIZE, label=ds + " pred")
        a.legend()

        fig.savefig("{}_chart_{}_epochs.jpg".format(ds, EPOCHS))

    print("Delta: ", time.time() - start_time)


if __name__ == "__main__":
    main()
