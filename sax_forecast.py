import cntk as C

import pandas as pd
import numpy as np
import time
from pandas_datareader import data as pd_data
import datetime
from saxpy.sax import sax_via_window

from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from saxpy.znorm import znorm


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

window_len = int(input("window_len: "))
word_len = int(input("word_len: "))
alphabet_len = int(input("alphabet_len: "))

# dat = np.array([5, 10, 10, 5, 20, 5, 0, 15, 15, 10, 0, 5, 10, 15, 20, 5, 5, 5, 0, 20, 10, 10, 15, 20, 20, 0, 0, 10, 10, 15])
# ts_data = pd_data.DataReader("SPY", "yahoo", "2018-12-01", datetime.datetime.now())
# sax_seq = ts_to_string(znorm(ts_data), cuts_for_asize(alphabet_len))

source = "weather_JAN.csv"
ts_data = pd.read_csv(source, index_col="date", parse_dates=["date"], dtype=np.float32)
sax_ret = sax_via_window(ts_data["temp"].values,
                         window_len,
                         word_len,
                         alphabet_size=alphabet_len,
                         nr_strategy="none",
                         z_threshold=0.01)


my_sax = {}
for k, v in sax_ret.items():
    for i in v:
        my_sax[i] = k

final_d = {"x": [], "y": []}
for k, v in my_sax.items():
    final_d["x"].append(np.float32("".join([str(ord(char) - 96) for char in v[:-1]])))
    final_d["y"].append(np.float32("".join([str(ord(char) - 96) for char in v[-1]])))


##### THIS FORMAT!!!
# result_x[0] = [1]         result_y[0] = 3
# result_x[1] = [1,4]       result_y[1] = 3
# result_x[2] = [1,4,2]     result_y[2] = 3
# result_x[3] = [1,4,2,2]   result_y[3] = 3
# result_x[4] = [1,4,2,2,4] result_y[4] = 3
#####

result_x = dict()
result_x["train"] = np.array(final_d["x"][:len(final_d["x"])-1000])
result_x["test"] = np.array(final_d["x"][len(final_d["x"])-1000:len(final_d["x"])-500])
result_x["val"] = np.array(final_d["x"][len(final_d["x"])-500:len(final_d["x"])])

result_y = dict()
result_y["train"] = np.array(final_d["y"][:len(final_d["y"])-1000])
result_y["test"] = np.array(final_d["y"][len(final_d["y"])-1000:len(final_d["y"])-500])
result_y["val"] = np.array(final_d["y"][len(final_d["y"])-500:len(final_d["y"])])

EPOCHS = 100
BATCH_SIZE = window_len * 10
H_DIMS = word_len

x = C.sequence.input_variable(1)
z = create_model(x, H_DIMS)
var_l = C.input_variable(1, dynamic_axes=z.dynamic_axes, name="y")
learning_rate = 0.005
lr_schedule = C.learning_parameter_schedule(learning_rate)
loss = C.squared_error(z, var_l)
error = C.squared_error(z, var_l)
momentum_schedule = C.momentum_schedule(0.9, minibatch_size=BATCH_SIZE)
learner = C.fsadagrad(z.parameters,
                      lr=lr_schedule,
                      momentum=momentum_schedule)
trainer = C.Trainer(z, (loss, error), [learner])

# training
loss_summary = []

start = time.time()
for epoch in range(0, EPOCHS):
    print("EPOCHS: ", EPOCHS)
    for x_batch, l_batch in next_batch(result_x, result_y, "train", BATCH_SIZE):
        trainer.train_minibatch({x: x_batch, var_l: l_batch})

    if epoch % (EPOCHS / 10) == 0:
        training_loss = trainer.previous_minibatch_loss_average
        loss_summary.append(training_loss)
        print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))

print("Training took {:.1f} sec".format(time.time() - start))

# Print the train, validation and test errors
for labeltxt in ["train", "val", "test"]:
    print("mse for {}: {:.6f}".format(labeltxt, get_mse(trainer, x, result_x, result_y, BATCH_SIZE, var_l, labeltxt)))

z.save("test.model")
