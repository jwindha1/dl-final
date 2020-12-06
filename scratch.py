import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from attention import *
from tcn import *

WINDOW_SIZE = 20
THRESHOLD = 100

with open('data.json') as f:
    data = tf.squeeze(tf.ragged.constant(json.load(f)), axis=1)

# split into train and test sets
train_size = int(data.shape[0] * 0.67)
train, test = data[0:train_size,:], data[train_size:data.shape[0],:]

# def create_data_labels(ts):
#     tuples, labels = [], []
#     for i in range(0, len(ts) - WINDOW_SIZE, WINDOW_SIZE):
#         tuples.append(ts[i : i + WINDOW_SIZE])
#         labels.append(1 if any(val > THRESHOLD for val in ts[i : i + WINDOW_SIZE]) else 0)
    
#     return tf.convert_to_tensor(tuples), tf.convert_to_tensor(labels)

def create_data_labels(ts):
    # rnn style
    # ts is a list of a time series
    tuples, labels = [], []
    for i in range(0, len(ts) - WINDOW_SIZE):
        tuples.append(ts[i : i + WINDOW_SIZE])
        labels.append(1 if ts[i + WINDOW_SIZE] >= 100 else 0)
    
    return tf.convert_to_tensor(tuples), tf.convert_to_tensor(labels)

# print(tf.concat(list(zip(*[create_data_labels(list(sample)) for sample in data]))[1], axis=0))

def split(data):
    return [tf.concat(lst, axis=0) for lst in zip(*[create_data_labels(list(sample)) for sample in data])]

X_train, y_train = split(train)
X_test, y_test = split(test)

print("input shapes", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = TCNModel()


# model = tf.keras.Sequential([
#         layers.Dense(2, activation="relu", name="layer1"),
#         layers.Dense(3, activation="relu", name="layer2"),
#         layers.Dense(2, name="layer3"),
#     ])


# model.compile(
#     optimizer=keras.optimizers.Adam(learning_rate=0.01),
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[keras.metrics.SparseCategoricalAccuracy()]
#     )

# model.fit(
#     X_train,
#     y_train,
#     batch_size=64,
#     epochs=2
# )

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    assert(train_inputs.shape[0] == train_labels.shape[0])
    total_num_examples = train_inputs.shape[0]
    shuffled_indices = tf.random.shuffle(tf.range(total_num_examples))
    shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
    shuffled_labels = tf.gather(train_labels, shuffled_indices)
    
    losses = []
    for start_ind in range(0, total_num_examples, model.batch_size):
        end_ind = min(start_ind + model.batch_size, total_num_examples)
        batched_inputs = shuffled_inputs[start_ind:end_ind, :]
        batched_labels = shuffled_labels[start_ind:end_ind]
        with tf.GradientTape() as tape:
            print(f'batched input {batched_inputs.shape}')
            print(f'batched labels {batched_labels}')
            logits = model.call(batched_inputs)
            print(f'logits! {logits.shape}')
            loss = model.loss(logits, batched_labels)
        # apply gradients to trainable variables after GradientTape
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(loss)
    
    return losses

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    logits = model.call(test_inputs)
    return model.accuracy(logits, test_labels)

train(model, X_train, y_train)

# model.summary()

# print("test acc: ", model.evaluate(X_test, y_test, batch_size=128)[1])