import cntk as C

import pandas as pd
from pandas_datareader import data
import numpy as np
import time
import datetime
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


def get_asset_data(source, contract, start_date, end_date):
    retry_cnt, max_num_retry = 0, 3
    while retry_cnt < max_num_retry:
        try:
            return pd_data.DataReader(contract, source, start_date, end_date)
        except Exception as e:
            print(e)
            retry_cnt += 1
            time.sleep(np.random.randint(1, 10))
    print("{} is not reachable".format(source))
    return []


def check_output(last_sax, window_len, word_len, alphabet_len):
    print("============================CHECK_OUTPUT============================")
    sax_ret = sax_via_window(last_sax,
                             window_len,
                             word_len,
                             alphabet_size=alphabet_len,
                             nr_strategy="none",
                             z_threshold=0.01)
    my_sax = dict()
    for k, v in sax_ret.items():
        for i in v:
            my_sax[i] = k
    for k, v in my_sax.items():
        print(k, v)
    print("max: ", max(last_sax))
    print("min: ", min(last_sax))


def prepare_data(window_len, word_len, alphabet_len, alpha_to_num, train_percent):
    source = input("Source (1=CSV,2=Finance): ")
    if source == "1":
        source = "weather_JAN.csv"
        ts_data = pd.read_csv(source, index_col="date", parse_dates=["date"], dtype=np.float32)
        sax_ret = sax_via_window(ts_data["temp"].values,
                                 window_len,
                                 word_len,
                                 alphabet_size=alphabet_len,
                                 nr_strategy="none",
                                 z_threshold=0.01)
    else:
        source = input("Remote Source (yahoo): ")
        if source == "":
            source = "yahoo"

        contract = input("Contract (SPY): ")
        if contract == "":
            contract = "SPY"

        start_date = input("Start Date (2000-01-01): ")
        if start_date == "":
            start_date = "2000-01-01"

        end_date = input("End Date (now): ")
        if end_date == "":
            end_date = datetime.datetime.now()

        ts_data = get_asset_data(source, contract, start_date, end_date)
        if "Close" in ts_data:
            close_tag = "Close"
        elif "close" in ts_data:
            close_tag = "close"
        else:
            return {"Error": "Couldn't find Close data."}
        sax_ret = sax_via_window(ts_data[close_tag].values,
                                 window_len,
                                 word_len,
                                 alphabet_size=alphabet_len,
                                 nr_strategy="none",
                                 z_threshold=0.01)

    print("LAST WINDOW ITEMS: ", ts_data[close_tag].values[-window_len:])

    my_sax = dict()
    for k, v in sax_ret.items():
        for i in v:
            my_sax[i] = k

    tmp_d = {"x": [], "y": []}
    for i in range(len(my_sax)):
        word = my_sax[i]
        if i < len(my_sax) - 2:
            pred = my_sax[i + 1][0]
            num_list = [np.float32(alpha_to_num[char][1]) for char in word]
            increment_list = []
            for num in num_list:
                increment_list.append(num)
                tmp_d["x"].append(np.array(increment_list))
                tmp_d["y"].append(np.array([np.float32(alpha_to_num[pred][1])]))

    print("LAST MY_SAX: ", my_sax[len(my_sax) - 1])

    # FORMAT:
    # result_x[0] = [1]         result_y[0] = 3
    # result_x[1] = [1,4]       result_y[1] = 3
    # result_x[2] = [1,4,2]     result_y[2] = 3
    # result_x[3] = [1,4,2,2]   result_y[3] = 3
    # result_x[4] = [1,4,2,2,4] result_y[4] = 3
    #####

    # Separate Dataset into train (80%), val (10%) and test (10%)
    pos_train = int(len(tmp_d["x"]) * train_percent)
    pos_train = int(pos_train / window_len) * window_len

    pos_val = len(tmp_d["x"][pos_train:]) / 2
    pos_val = pos_train + int(pos_val / window_len) * window_len

    pos_test = pos_val

    result_x = dict()
    result_x["train"] = tmp_d["x"][:pos_train]
    result_x["val"] = tmp_d["x"][pos_train:pos_val]
    result_x["test"] = tmp_d["x"][pos_test:]

    result_y = dict()
    result_y["train"] = np.array(tmp_d["y"][:pos_train])
    result_y["val"] = np.array(tmp_d["y"][pos_train:pos_val])
    result_y["test"] = np.array(tmp_d["y"][pos_val:])

    return result_x, result_y


def main():
    window_len = int(input("window_len: "))
    word_len = int(input("word_len: "))
    alphabet_len = int(input("alphabet_len: "))

    train_percent = float(input("Train %: "))

    epochs = input("Epochs: ")
    if not epochs == "":
        epochs = int(epochs)
    else:
        epochs = 100

    # batch_size = window_len * (word_len - 1)
    batch_size = int(input("Batch size: "))
    h_dims = word_len + 1

    alpha_to_num_step = float(1 / alphabet_len)
    alpha_to_num_shift = float(alpha_to_num_step / 2)

    # Dict = [floor, point, celling]
    alpha_to_num = dict()
    for i in range(alphabet_len):
        step = (alpha_to_num_step * i)
        alpha_to_num[chr(97 + i)] = [step,
                                     step + alpha_to_num_shift,
                                     step + alpha_to_num_step]

    model_file = "{}_{}_{}_{}.model".format(window_len, word_len, alphabet_len, epochs)
    if input("Change model name [{}]? ".format(model_file)) == "y":
        model_file = input("Model filename: ")

    x, y = prepare_data(window_len, word_len, alphabet_len, alpha_to_num, train_percent)

    if input("Training? ") == "y":
        input_node = C.sequence.input_variable(1)
        z = create_model(input_node, h_dims)
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
            for x_batch, l_batch in next_batch(x, y, "train", batch_size):
                trainer.train_minibatch({input_node: x_batch, var_l: l_batch})

            if epoch % (epochs / 10) == 0:
                training_loss = trainer.previous_minibatch_loss_average
                loss_summary.append(training_loss)
                print("epoch: {}, loss: {:.4f} [time: {:.1f}s]".format(epoch, training_loss, time.time() - start))

        print("Training took {:.1f} sec".format(time.time() - start))

        # Print the train, validation and test errors
        for label_txt in ["train", "val", "test"]:
            print("mse for {}: {:.6f}".format(label_txt, get_mse(trainer,
                                                                 input_node,
                                                                 x,
                                                                 y,
                                                                 batch_size,
                                                                 var_l,
                                                                 label_txt)))
        z.save(model_file)
    else:
        z = C.load_model(model_file)
        input_node = C.logging.find_all_with_name(z, "")[-1]

    # Print out all layers in the model
    print("Loading {} and printing all nodes:".format(model_file))
    node_outputs = C.logging.find_all_with_name(z, "")
    for n in node_outputs:
        print("  {}".format(n))

    # predict
    results = []
    for j, ds in enumerate(["val", "test"]):
        fig = plt.figure()
        chart = fig.add_subplot(2, 1, 1)
        results = []
        for x_batch, y_batch in next_batch(x, y, ds, batch_size):
            pred = z.eval({input_node: x_batch})
            results.extend(pred[:, 0])

        # chart.plot((result_y[ds]).flatten(), label=ds + " raw")
        # chart.plot(np.array(results), label=ds + " pred")

        last_p_y = []
        for idx, i in enumerate(y[ds]):
            if (idx + 1) % (word_len + 1) == 0:
                last_p_y.append(i)

        chart.plot(np.array(last_p_y).flatten(), label=ds + " raw")

        last_p_result = []
        for idx, i in enumerate(results):
            if (idx + 1) % (word_len + 1) == 0:
                alpha_list = sorted(alpha_to_num)
                a = "a"
                for a in alpha_list[::-1]:
                    if i >= alpha_to_num[a][0]:
                        break
                last_p_result.append(alpha_to_num[a][1])

        chart.plot(np.array(last_p_result), label=ds + " pred")
        chart.legend()

        fig.savefig("{}_chart_{}_epochs.jpg".format(ds, epochs))

        correct_pred = dict()
        for idx, _ in enumerate(last_p_y):
            print("{}: {} == {} ({})".format(idx,
                                             last_p_result[idx],
                                             float(last_p_y[idx][0]),
                                             last_p_result[idx] - float(last_p_y[idx][0])))
            alpha_list = sorted(alpha_to_num)
            for pred_a in alpha_list[::-1]:
                if last_p_result[idx] >= alpha_to_num[pred_a][0]:
                    pred_l_num = ord(pred_a)
                    for y_a in alpha_list[::-1]:
                        if float(last_p_y[idx][0]) >= alpha_to_num[y_a][0]:
                            stp = abs(ord(y_a) - pred_l_num)
                            print("stp: ", stp)
                            if stp not in correct_pred:
                                correct_pred[stp] = 1
                            else:
                                correct_pred[stp] += 1
                            break
                    break

        for k, v in correct_pred.items():
            print("Set({}) Delta[{}]: {}/{} = {:.4f}".format(ds,
                                                             k,
                                                             v,
                                                             len(last_p_y),
                                                             float(v / len(last_p_y))))
        print("len(last_p_y): ", len(last_p_y))
        print("len(last_p_result): ", len(last_p_result))

        check_output(x["test"][-window_len:], window_len, word_len, alphabet_len)

    for k, v in alpha_to_num.items():
        print(k, v)

    return x, y, results


if __name__ == '__main__':
    r_x, r_y, r_test_pred = main()
