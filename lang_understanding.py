import requests
import os

import math
import numpy as np

import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components


def download(url, filename):
    """ utility function to download a file """
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        for data in response.iter_content():
            handle.write(data)


def create_model(emb_dim, hidden_dim, num_labels):
    with C.layers.default_options(initial_state=0.1):
        return C.layers.Sequential([
            C.layers.Embedding(emb_dim, name='embed'),
            C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False),
            C.layers.Dense(num_labels, name='classify')
        ])


def create_reader(in_path, vocab_size, num_intents, num_labels, is_training):
    return C.io.MinibatchSource(C.io.CTFDeserializer(in_path, C.io.StreamDefs(
         query         = C.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
         intent        = C.io.StreamDef(field='S1', shape=num_intents, is_sparse=True),
         slot_labels   = C.io.StreamDef(field='S2', shape=num_labels,  is_sparse=True)
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)


def create_criterion_function_preferred(model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)


def train(x, y, reader, model_func, max_epochs=10, task='slot_tagging'):
    print("Training...")

    # Instantiate the model function; x is the input (feature) variable
    model = model_func(x)

    # Instantiate the loss and error function
    loss, label_error = create_criterion_function_preferred(model, y)

    # training config
    epoch_size = 18000        # 18000 samples is half the dataset size
    minibatch_size = 70

    # LR schedule over epochs
    # In CNTK, an epoch is how often we get out of the minibatch loop to
    # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
    lr_per_sample = [3e-4]*4+[1.5e-4]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_parameter_schedule(lr_per_minibatch, epoch_size=epoch_size)

    # Momentum schedule
    momentums = C.momentum_schedule(0.9048374180359595, minibatch_size=minibatch_size)

    # We use a the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.adam(parameters=model.parameters,
                     lr=lr_schedule,
                     momentum=momentums,
                     gradient_clipping_threshold_per_sample=15,
                     gradient_clipping_with_truncation=True)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)

    # Uncomment below for more detailed logging
    # progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs)

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    # Assign the data fields to be read from the input
    if task == 'slot_tagging':
        data_map={x: reader.streams.query, y: reader.streams.slot_labels}
    else:
        data_map={x: reader.streams.query, y: reader.streams.intent}

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map= data_map)  # fetch minibatch
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()

    return


def evaluate(x, y, reader, model_func, task='slot_tagging'):

    print("Evaluating...")

    # Instantiate the model function; x is the input (feature) variable
    model = model_func(x)

    # Create the loss and error functions
    loss, label_error = create_criterion_function_preferred(model, y)

    # process minibatches and perform evaluation
    progress_printer = C.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)

    # Assign the data fields to be read from the input
    if task == 'slot_tagging':
        data_map = {x: reader.streams.query, y: reader.streams.slot_labels}
    else:
        data_map = {x: reader.streams.query, y: reader.streams.intent}

    evaluator = None
    while True:
        minibatch_size = 500
        data = reader.next_minibatch(minibatch_size, input_map=data_map)
        if not data:
            break

        evaluator = C.eval.Evaluator(loss, progress_printer)
        evaluator.test_minibatch(data)

    if evaluator:
        evaluator.summarize_test_progress()
    else:
        print("Error: evaluator is None")


def main():
    locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

    data = {
      'train': {'file': 'atis.train.ctf', 'location': 0},
      'test': {'file': 'atis.test.ctf', 'location': 0},
      'query': {'file': 'query.wl', 'location': 1},
      'slots': {'file': 'slots.wl', 'location': 1},
      'intent': {'file': 'intent.wl', 'location': 1}
    }

    for item in data.values():
        location = locations[item['location']]
        path = os.path.join('..', location, item['file'])
        if os.path.exists(path):
            print("Reusing locally cached:", item['file'])
            # Update path
            item['file'] = path
        elif os.path.exists(item['file']):
            print("Reusing locally cached:", item['file'])
        else:
            print("Starting download:", item['file'])
            url = "https://github.com/Microsoft/CNTK/blob/release/2.6/%s/%s?raw=true"%(location, item['file'])
            download(url, item['file'])
            print("Download completed")

    # number of words in vocab, slot labels, and intent labels
    vocab_size = 943
    num_labels = 129
    num_intents = 26

    # model dimensions
    input_dim  = vocab_size
    label_dim  = num_labels
    emb_dim    = 150
    hidden_dim = 300

    # Create the containers for input feature (x) and the label (y)
    x_input = C.sequence.input_variable(vocab_size)
    y_input = C.sequence.input_variable(num_labels)

    model_file = input("Model file name: ")
    if not os.path.exists(model_file):
        z = create_model(emb_dim, hidden_dim, num_labels)

        # peek
        reader = create_reader(data['train']['file'], vocab_size, num_intents, num_labels, is_training=True)
        train(x_input, y_input, reader, z)

        reader = create_reader(data['test']['file'], vocab_size, num_intents, num_labels, is_training=False)
        evaluate(x_input, y_input, reader, z)

        z.save(model_file)
    else:
        z = C.load_model(model_file)

    # load dictionaries
    query_wl = [line.rstrip('\n') for line in open(data['query']['file'])]
    slots_wl = [line.rstrip('\n') for line in open(data['slots']['file'])]
    query_dict = {query_wl[i]: i for i in range(len(query_wl))}
    slots_dict = {slots_wl[i]: i for i in range(len(slots_wl))}

    # let's run a sequence through
    seq = 'BOS flights from new york to seattle EOS'
    w = [query_dict[w] for w in seq.split()]
    print(w)
    onehot = np.zeros([len(w), len(query_dict)], np.float32)
    for t in range(len(w)):
        onehot[t, w[t]] = 1

    # x = C.sequence.input_variable(vocab_size)
    pred = z(x_input).eval({x_input: [onehot]})[0]
    print(pred.shape)
    best = np.argmax(pred, axis=1)
    print(best)
    print(list(zip(seq.split(), [slots_wl[s] for s in best])))


if __name__ == '__main__':
    main()
