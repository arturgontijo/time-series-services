import cntk as C

import pandas as pd
import numpy as np
import time
from pandas_datareader import data as pd_data

from saxpy.sax import sax_via_window

import os
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def next_batch(x, y, ds, batch_size):
    """get the next batch for training"""
    def as_batch(data, start, count):
        return data[start:start + count]
    for i in range(0, len(x[ds]), batch_size):
        yield as_batch(x[ds], i, batch_size), as_batch(y[ds], i, batch_size)


# validate
def get_mse(trainer_local, x_label, x_local, y_local, batch_size, l_label, label_txt):
    result = 0.0
    for x1, y1 in next_batch(x_local, y_local, label_txt, batch_size):
        eval_error = trainer_local.test_minibatch({x_label: x1, l_label: y1})
        result += eval_error
    return result/len(x_local[label_txt])


def create_model(x_local, h_dims):
    """Create the model for time series prediction"""
    with C.layers.default_options(initial_state=0.1):
        m = C.layers.Recurrence(C.layers.LSTM(h_dims))(x_local)
        m = C.sequence.last(m)
        m = C.layers.Dropout(0.2)(m)
        m = C.layers.Dense(1)(m)
        return m


def main():
    window_len = int(input("window_len: "))
    word_len = int(input("word_len: "))
    alphabet_len = int(input("alphabet_len: "))

    alpha_to_num_step = float(1 / alphabet_len)
    alpha_to_num_shift = float(alpha_to_num_step / 2)

    alpha_to_num = dict()
    for i in range(alphabet_len):
        alpha_to_num[chr(97+i)] = (alpha_to_num_step * i) + alpha_to_num_shift

    source = "weather_JAN.csv"
    ts_data = pd.read_csv(source, index_col="date", parse_dates=["date"], dtype=np.float32)
    sax_ret = sax_via_window(ts_data["temp"].values,
                             window_len,
                             word_len,
                             alphabet_size=alphabet_len,
                             nr_strategy="none",
                             z_threshold=0.01)

    my_sax = dict()
    for k, v in sax_ret.items():
        for i in v:
            my_sax[i] = k

    tmp_d = {"x": [], "y": []}
    for k, v in my_sax.items():
        num_list = [np.float32(alpha_to_num[char]) for char in v[:-1]]
        increment_list = []
        for num in num_list:
            increment_list.append(num)
            tmp_d["x"].append(np.array(increment_list))
            tmp_d["y"].append(np.array([np.float32(alpha_to_num[char]) for char in v[-1]]))

    # FORMAT:
    # result_x[0] = [1]         result_y[0] = 3
    # result_x[1] = [1,4]       result_y[1] = 3
    # result_x[2] = [1,4,2]     result_y[2] = 3
    # result_x[3] = [1,4,2,2]   result_y[3] = 3
    # result_x[4] = [1,4,2,2,4] result_y[4] = 3
    #####

    # Separate Dataset into train (80%), val (10%) and test (10%)
    pos_train = len(tmp_d["x"]) * 0.8
    pos_val = pos_train + len(tmp_d["x"]) * 0.1
    pos_test = pos_val + len(tmp_d["x"])

    result_x = dict()
    result_x["train"] = tmp_d["x"][:pos_train]
    result_x["val"] = tmp_d["x"][pos_train:pos_val]
    result_x["test"] = tmp_d["x"][pos_test:len(tmp_d["x"])]

    result_y = dict()
    result_y["train"] = np.array(tmp_d["y"][:pos_train])
    result_y["val"] = np.array(tmp_d["y"][pos_train:pos_val])
    result_y["test"] = np.array(tmp_d["y"][pos_val:len(tmp_d["y"])])

    batch_size = window_len * (word_len - 1)
    h_dims = word_len

    epochs = input("Epochs: ")
    if not epochs == "":
        epochs = int(epochs)
    else:
        epochs = 100

    start_time = time.time()

    model_file = "{}_epochs.model".format(epochs)

    if not os.path.exists(model_file):
        x = C.sequence.input_variable(1)
        z = create_model(x, h_dims)
        var_l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")
        learning_rate = 0.005
        lr_schedule = C.learning_parameter_schedule(learning_rate)
        loss = C.squared_error(z, var_l)
        error = C.squared_error(z, var_l)
        momentum_schedule = C.momentum_schedule(0.9, minibatch_size=batch_size)
        learner = C.fsadagrad(z.parameters,
                              lr=lr_schedule,
                              momentum=momentum_schedule)
        trainer = C.Trainer(z, (loss, error), [learner])

        # training
        loss_summary = []

        start = time.time()
        for epoch in range(0, epochs):
            for x_batch, l_batch in next_batch(result_x, result_y, "train", batch_size):
                trainer.train_minibatch({x: x_batch, var_l: l_batch})

            if epoch % (epochs / 10) == 0:
                training_loss = trainer.previous_minibatch_loss_average
                loss_summary.append(training_loss)
                print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))

        print("Training took {:.1f} sec".format(time.time() - start))

        # Print the train, validation and test errors
        for label_txt in ["train", "val", "test"]:
            print("mse for {}: {:.6f}".format(label_txt, get_mse(trainer, x, result_x, result_y, batch_size, var_l, label_txt)))

        z.save(model_file)

    else:
        z = C.load_model(model_file)
        x = C.logging.find_all_with_name(z, "")[-1]

    # Print out all layers in the model
    print("Loading {} and printing all nodes:".format(model_file))
    node_outputs = C.logging.find_all_with_name(z, "")
    for n in node_outputs:
        print("  {}".format(n))

    results = []
    # predict
    # f, a = plt.subplots(2, 1, figsize=(12, 8))
    for j, ds in enumerate(["val", "test"]):
        fig = plt.figure()
        a = fig.add_subplot(2, 1, 1)
        results = []
        for x_batch, y_batch in next_batch(result_x, result_y, ds, batch_size):
            pred = z.eval({x: x_batch})
            results.extend(pred[:, 0])
        # because we normalized the input data we need to multiply the prediction
        # with SCALER to get the real values.
        a.plot((result_y[ds]).flatten(), label=ds + " raw")
        a.plot(np.array(results), label=ds + " pred")
        a.legend()

        fig.savefig("{}_chart_{}_epochs.jpg".format(ds, epochs))

    print("Delta: ", time.time() - start_time)

    return result_x, result_y, results


if __name__ == '__main__':
    r_x, r_y, r_test_pred = main()
