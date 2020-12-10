import json
import tensorflow as tf
import matplotlib.pyplot as plt
from attention import *
from tcn import *
import numpy as np
from sklearn.decomposition import PCA

THRESHOLD = 100

def load_data(filename='data.json'):
    '''
    Reads data and splits into test and train 
    :param: filename: the data file name
    :return: two 2D ragged tensors where the first dimension is the participant
    and the second dimension is the time series data
    '''
    with open(filename) as f:
        raw_data = json.load(f)
    
    data = tf.squeeze(tf.ragged.constant(raw_data), axis=2)
    
    train_size = int(data.shape[0] * 0.67)
    train_data = data[0:train_size,:]
    test_data = data[train_size:data.shape[0],:]
    return train_data, test_data

def create_data_labels(ts, window_size):
    '''
    Splits a time series into periods (rnn-style) and generates labels
    :param: ts: an iterable of floats representing time series values
    :param: window_size: an int representing how many data will be in a period
    :return: two 2D tensors where the first is num_periods_for_timeseries x 
    window_size representing the time series and the second is 
    num_periods_for_timeseries x 2 representing the one-hot vector of labels
    '''
    tuples, labels = [], []
    for i in range(0, len(ts) - window_size):
        tuples.append(ts[i : i + window_size])
        labels.append([0, 1] if ts[i + window_size] >= 100 else [1, 0])
    
    return tf.convert_to_tensor(tuples), tf.convert_to_tensor(labels)

def split(data, window_size=20):
    '''
    Shapes and concatenates all time series' data and labels (removes 
    participant axis)
    :param: data: a ragged tensor of dims num_participants x None, where the 
    second dim is the time series for the participant
    :param: window_size: an int representing how many data will be in a period
    :return: a list of two elements representing the X and y for all 
    participants added together (num_periods_for_all_timeseries x window_size
    for X and num_periods_for_all_timeseries x 2 for y)
    '''
    return [tf.concat(lst, axis=0) for lst in zip(*[create_data_labels(list(sample), window_size) for sample in data])]

def get_data(window_size=20, filename='data.json'):
    '''
    Loads and splits data into training and testing data and labels
    :param: window_size: an int representing how many data will be in a period
    :filename: the data file name
    :return: training data and labels and testing data and labels, with shapes
    num_periods_for_all_timeseries x window_size for data and 
    num_periods_for_all_timeseries x 2 for labels
    '''
    train_data, test_data = load_data(filename=filename)
    X_train, y_train = split(train_data, window_size=window_size)
    X_test, y_test = split(test_data, window_size=window_size)
    return X_train, y_train, X_test, y_test

def get_data_pca(window_size=20, filename='data_multiple_features.json'):
    '''
    Loads and splits data into training and testing data and labels using 
    multiple features and PCA to extract data values
    :param: window_size: an int representing how many data will be in a period
    :filename: the data file name
    :return: training data and labels and testing data and labels, with shapes
    num_periods_for_all_timeseries x window_size for data and 
    num_periods_for_all_timeseries x 2 for labels
    '''
    with open('data.json') as f:
        non_pca_data = tf.squeeze(tf.ragged.constant(json.load(f)), axis=2)

    with open(filename) as f:
        raw_data = json.load(f)
    
    pca = PCA(n_components=1)
    pca_train_data = np.array([item for sublist in raw_data for item in sublist])
    pca.fit(pca_train_data)
    data = tf.squeeze(tf.ragged.constant([pca.transform(individ) for individ in raw_data]), axis=2)

    all_tuples, all_labels = [], []
    for ts, non_pca in zip(data, non_pca_data):
        tuples, labels = [], []
        for i in range(0, len(ts) - window_size):
            tuples.append(ts[i : i + window_size])
            labels.append([0, 1] if non_pca[i + window_size] >= 100 else [1, 0])
        all_tuples.append(tuples)
        all_labels.append(labels)

    train_size = int(len(all_tuples) * 0.67)
    train_X, test_X = tf.concat(all_tuples[:train_size], axis=0), tf.concat(all_tuples[train_size:], axis=0)
    train_y, test_y = tf.concat(all_labels[:train_size], axis=0), tf.concat(all_labels[train_size:], axis=0)
    return train_X, train_y, test_X, test_y

def train(model, train_inputs, train_labels):
    '''
    Trains the model in one epoch in batches and returns all losses
    :param: model: a TCNModel
    :param: train_inputs: a 2D tensor of shape num_periods_for_all_timeseries 
    x window_size
    :param: train_labels: a 2D tensor of shape num_periods_for_all_timeseries 
    x 2
    :return: a list of the loss for each batch
    '''
    assert(train_inputs.shape[0] == train_labels.shape[0])
    total_num_examples = train_inputs.shape[0]
    shuffled_indices = tf.random.shuffle(tf.range(total_num_examples))
    shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
    shuffled_labels = tf.gather(train_labels, shuffled_indices)
    
    losses = []
    for start_ind in range(0, total_num_examples, model.batch_size):
        if start_ind + model.batch_size > total_num_examples: 
            break
        end_ind = start_ind + model.batch_size
        batched_inputs = shuffled_inputs[start_ind:end_ind, :]
        batched_labels = shuffled_labels[start_ind:end_ind]
        with tf.GradientTape() as tape:
            logits = model.call(batched_inputs)
            loss = model.loss(logits, batched_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(loss)
    
    return losses

def test(model, test_inputs, test_labels):
    '''
    Tests the model in batches and returns all accuracies
    :param: model: a TCNModel
    :param: test_inputs: a 2D tensor of shape num_periods_for_all_timeseries 
    x window_size
    :param: test_labels: a 2D tensor of shape num_periods_for_all_timeseries 
    x 2
    :return: a list of the average accuracy for each batch
    '''

    assert(test_inputs.shape[0] == test_labels.shape[0])
    total_num_examples = test_inputs.shape[0]
    shuffled_indices = tf.random.shuffle(tf.range(total_num_examples))
    shuffled_inputs = tf.gather(test_inputs, shuffled_indices)
    shuffled_labels = tf.gather(test_labels, shuffled_indices)
    
    accs = []
    for start_ind in range(0, total_num_examples, model.batch_size):
        if start_ind + model.batch_size > total_num_examples: 
            break
        end_ind = start_ind + model.batch_size
        batched_inputs = shuffled_inputs[start_ind:end_ind, :]
        batched_labels = shuffled_labels[start_ind:end_ind]
        logits = model.call(batched_inputs)
        acc = model.accuracy(logits, batched_labels)
        accs.append(acc.numpy())

    return accs