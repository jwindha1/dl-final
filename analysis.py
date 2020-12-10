import json
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from main import *
from attention import *
from tcn import *
from scipy.signal import lfilter

def visualize_time_series():
    '''
    Creates a plot of a randomly selected time series to show what our 
    raw data looks like
    '''

    with open('data.json') as data:
        raw_data = json.load(data)
    y = np.array(raw_data[5][0])
    x = np.arange(len(y))

    # Threshold above which the line should be red
    threshold = 100
    lower = np.ma.masked_where(y < threshold, y)
    upper = np.ma.masked_where(y > threshold, y)

    fig, ax = plt.subplots()
    ax.plot(x, upper, color="#31D9BD", label="Non-Stressed Labeling")
    ax.plot(x, lower, color="#F03681", label="Stressed Labeling")
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='best')

    plt.xlabel("Time (s)")
    plt.ylabel("Heart Rate (beats/min)")
    plt.show()

def compare_learning_rates(plot=True):
    '''
    Plots loss over training for multiple models with different learning rates
    :param: plot: whether or not to display the plot
    :return: a list of the different learning rates tested, a list of lists 
    of losses for each learning rate
    '''
    X_train, y_train, X_test, y_test = get_data()

    rates = [0.0005, 0.001, 0.005, 0.01, 0.05]

    colors = ["#F03681", "#456BDD", "#FFC13A", "#7B43DE", "#31D9BD"]
    all_losses = []
    all_losses_unsmooth = []
    for rate in rates:
        model = TCNModel(rate)
        losses = train(model, X_train, y_train)
        smoothness = 100
        yy = lfilter([1.0 / smoothness] * smoothness, 1, losses)
        all_losses.append(yy[20:])
        all_losses_unsmooth.append(losses)

    x = range(max([len(losses) for losses in all_losses]))
    for losses, rate, color in zip(all_losses, rates, colors):
        plt.plot(x, losses, label=rate, color=color)

    plt.yscale("log")
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend(title="Learning Rate")
    if plot: plt.show()

    return rates, all_losses_unsmooth


def compare_window_sizes():
    '''
    Trains a model and then plots the distribution of accuracies for each batch
    in testing using various window sizes
    :return: a list of the different window sizes tested, a list of lists 
    of test accuracies for each window size
    '''
    window_sizes = [1, 2, 10, 20, 30, 50, 100]
    all_accs = []
    for window_size in window_sizes:
        X_train, y_train, X_test, y_test = get_data(window_size=window_size)
        model = TCNModel()
        train(model, X_train, y_train)
        accs = test(model, X_test, y_test)
        all_accs.append(accs)
    
    return window_sizes, all_accs

def compare_pca_nonpca():
    '''
    Trains and tests the model 10 times with and without using PCA
    :return: two lists, each of ten lists of average model accuracies
    '''
    X_train_pca, y_train_pca, X_test_pca, y_test_pca = get_data_pca()
    X_train, y_train, X_test, y_test = get_data()
    all_accs_pca, all_accs = [], []
    for _ in range(10):
        model_pca = TCNModel()
        train(model_pca, X_train_pca, y_train_pca)
        accs_pca = test(model_pca, X_test_pca, y_test_pca)
        all_accs_pca.append(sum(accs_pca) / len(accs_pca))

        model = TCNModel()
        train(model, X_train, y_train)
        accs = test(model, X_test, y_test)
        all_accs.append(sum(accs) / len(accs))
    
    return all_accs_pca, all_accs

def all_experiments():
    '''
    Runs several experiments to see how various hyperparameters perform
    '''
    rates, all_losses = compare_learning_rates(plot=False)
    print(f'learning rate \t mean loss \t final loss')
    for rate, losses in zip(rates, all_losses):
        print(f'{rate} \t {round(float(sum(losses)) / len(losses), 3)} \t {round(float(losses[-1]), 3)}')

    window_sizes, all_accs = compare_window_sizes()
    print(f'window size \t mean accuracy \t max accuracy')
    for window_size, accs in zip(window_sizes, all_accs):
        print(f'{window_size} \t {round(float(sum(accs)) / len(accs), 3)} \t {round(float(max(accs)), 3)}')
    
    all_accs_pca, all_accs = compare_pca_nonpca()
    print(f'with pca? \t mean accuracy \t max accuracy')
    print(f'yes \t {round(sum(all_accs_pca) / len(all_accs_pca), 3)} \t {round(max(all_accs_pca), 3)}')
    print(f'no \t {round(sum(all_accs) / len(all_accs), 3)} \t {round(max(all_accs), 3)}')

# all_experiments()