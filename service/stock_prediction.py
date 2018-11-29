from __future__ import print_function

import cntk as C

import logging
import datetime
import numpy as np
import pandas as pd
# pip install git+https://github.com/pydata/pandas-datareader.git
from pandas_datareader import data
import time
import traceback


logging.basicConfig(level=10, format="%(asctime)s - [%(levelname)8s] - %(name)s - %(message)s")
log = logging.getLogger("stock_prediction")

# fix a random seed for CNTK components
C.cntk_py.set_fixed_random_seed(1)
# default='warn'
pd.options.mode.chained_assignment = None
# Set a random seed
np.random.seed(123)


class StockPrediction:
    def __init__(self, source, contract, start, end, target_date):
        self.source = source
        self.contract = contract
        self.start = start
        self.end = end
        self.target_date = target_date

        self.response = ""

    @staticmethod
    def _create_model(net_input, num_output_classes, num_hidden_layers, hidden_layers_dim):
        h = net_input
        with C.layers.default_options(init=C.glorot_uniform()):
            for i in range(num_hidden_layers):
                h = C.layers.Dense(hidden_layers_dim,
                                   activation=C.relu)(h)
            return C.layers.Dense(num_output_classes, activation=None)(h)

    @staticmethod
    # Defines a utility that prints the training progress
    def _print_training_progress(trainer, mb, frequency, verbose=1):
        training_loss = "NA"
        eval_error = "NA"
        if mb % frequency == 0:
            training_loss = trainer.previous_minibatch_loss_average
            eval_error = trainer.previous_minibatch_evaluation_average
            if verbose:
                print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error * 100))
        return mb, training_loss, eval_error

    def _get_stock_data(self):
        retry_cnt, max_num_retry = 0, 3

        while retry_cnt < max_num_retry:
            try:
                end = datetime.datetime.now()
                return data.DataReader(self.contract, self.source, "2000-01-01", end.date())
            except:
                retry_cnt += 1
                time.sleep(np.random.randint(1, 10))

        log.error("{} is not reachable".format(self.source))
        return []

    def stock_prediction(self):
        stock_data = self._get_stock_data()
        # Feature name list
        predictor_names = []

        if "Close" in stock_data and "Volume" in stock_data:
            close_tag = "Close"
            volume_tag = "Volume"
        elif "close" in stock_data and "volume" in stock_data:
            close_tag = "close"
            volume_tag = "volume"
        else:
            return {
                "Error": "Couldn't find Close|Volume data"
            }

        # Compute price difference as a feature
        stock_data["diff"] = np.abs(
            (stock_data[close_tag] - stock_data[close_tag].shift(1)) / stock_data[close_tag]).fillna(0)
        predictor_names.append("diff")

        # Compute the volume difference as a feature
        stock_data["v_diff"] = np.abs(
            (stock_data[volume_tag] - stock_data[volume_tag].shift(1)) / stock_data[volume_tag]).fillna(0)
        predictor_names.append("v_diff")

        # Compute the stock being up (1) or down (0) over different day offsets compared to current closing price
        num_days_back = 8
        for i in range(1, num_days_back + 1):
            # i: number of look back days
            stock_data["p_" + str(i)] = np.where(stock_data[close_tag] > stock_data[close_tag].shift(i), 1, 0)
            predictor_names.append("p_" + str(i))

        stock_data["next_day"] = np.where(stock_data[close_tag].shift(-1) > stock_data[close_tag], 1, 0)

        # The label must be one-hot encoded
        stock_data["next_day_opposite"] = np.where(stock_data["next_day"] == 1, 0, 1)

        # Establish the start and end date of our training timeseries
        training_data = stock_data[self.start:self.end]

        training_features = np.asarray(training_data[predictor_names], dtype="float32")
        training_labels = np.asarray(training_data[["next_day", "next_day_opposite"]], dtype="float32")

        # Lets build the network
        input_dim = 2 + num_days_back
        # Remember we need to have 2 since we are trying to classify if the market goes up or down 1 hot encoded
        num_output_classes = 2
        num_hidden_layers = 2
        hidden_layers_dim = 2 + num_days_back
        input_dynamic_axes = [C.Axis.default_batch_axis()]
        net_input = C.input_variable(input_dim, dynamic_axes=input_dynamic_axes)
        label = C.input_variable(num_output_classes, dynamic_axes=input_dynamic_axes)

        z = self._create_model(net_input, num_output_classes, num_hidden_layers, hidden_layers_dim)
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

        l_training_features = len(training_features)
        training_features = training_features[:l_training_features - (l_training_features % num_minibatches)]
        l_training_labels = len(training_labels)
        training_labels = training_labels[:l_training_labels - (l_training_labels % num_minibatches)]

        # Train our neural network
        tf = np.split(training_features, num_minibatches)
        tl = np.split(training_labels, num_minibatches)

        for i in range(num_minibatches * num_passes):  # multiply by the
            features = np.ascontiguousarray(tf[i % num_minibatches])
            labels = np.ascontiguousarray(tl[i % num_minibatches])

            # Specify the mapping of input variables in the model to actual minibatch data to be trained with
            trainer.train_minibatch({net_input: features, label: labels})
            batchsize, loss, error = self._print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
            if not (loss == "NA" or error == "NA"):
                plotdata["batchsize"].append(batchsize)
                plotdata["loss"].append(loss)
                plotdata["error"].append(error)

        # Now that we have trained the net, and we will do out of sample test to see how we did.
        # and then more importantly analyze how that set did

        test_data = stock_data[self.target_date]

        test_features = np.ascontiguousarray(test_data[predictor_names], dtype="float32")
        test_labels = np.ascontiguousarray(test_data[["next_day", "next_day_opposite"]], dtype="float32")

        avg_error = trainer.test_minibatch({net_input: test_features, label: test_labels})
        print("Average error: {0:2.2f}%".format(avg_error * 100))

        out = C.softmax(z)
        predicted_label_prob = out.eval({net_input: test_features})
        test_data["p_up"] = pd.Series(predicted_label_prob[:, 0], index=test_data.index)
        test_data["p_down"] = predicted_label_prob[:, 1]
        test_data['positions'] = test_data["long_entries"].fillna(0) + test_data["short_entries"].fillna(0)
        test_data["pnl"] = test_data[close_tag].diff().shift(-1).fillna(0) * test_data["positions"] / np.where(
            test_data[close_tag] != 0, test_data[close_tag], 1)
        test_data["perc"] = (test_data[close_tag] - test_data[close_tag].shift(1)) / test_data[close_tag].shift(1)

        return {
            "p_up": test_data["p_up"],
            "p_down": test_data["p_down"]
        }
