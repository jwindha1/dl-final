from __future__ import absolute_import
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import numpy as np
import random
import math
from attention import Attention


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 100  # Batch size for training/testing
        self.loss_list = []  # Append losses to this list in training so you can visualize loss vs time in main
        self.num_classes = 2  # Number of classes of objects
        self.padding = 'SAME'  # Padding type
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Stochastic Gradient Descent

        self.conv1 = tf.keras.layers.Conv1D(filters =32, strides= 1, padding = "causal", dilation = 1)
        self.att1= Attention(32)
        self.conv2 = tf.keras.layers.Conv1D(filters=16, strides=1, padding="causal", dilation=2)
        self.att1 = Attention(16)
        self.conv3 = tf.keras.layers.Conv1D(filters=8, strides=1, padding="causal", dilation=4)
        self.att1 = Attention(8)
        self.conv4 = tf.keras.layers.Conv1D(filters=4, strides=1, padding="causal", dilation=8)
        self.att1 = Attention(4)
        self.conv5 = tf.keras.layers.Conv1D(filters=2, strides=1, padding="causal", dilation=16)
        self.att1= Attention(2)
        self.conv6 = tf.keras.layers.Conv1D(filters=1, strides=1, padding="causal", dilation=32)
        self.acrross_attn =  Attention(1)
        self.dense = tf.keras.layers.Dense(units = 1)


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        c1 = self.conv1(inputs)
        gamma1 = self.att1.call(c1)
        c2 = self.conv2(c1)
        gamma2 = self.att2.call(c2)
        c3 = self.conv3(c2)
        gamma3 = self.att3.call(c3)
        c4 = self.conv4(c3)
        gamma4 = self.att4.call(c4)
        c5 = self.conv5(c4)
        gamma5 = self.att5.call(c5)
        c6 = self.conv6(c5)
        gamma6 = self.att6.call(c6)

        within_attention  = tf.concat([gamma1, gamma2, gamma3, gamma4, gamma5, gamma6], axis = 0)
        final_gamma = self.acrross_attn.call(within_attention)
        output = self.dense(final_gamma)

        return output











        return lj3

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))
        return total_loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


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
    index_list = list(range(0, tf.shape(train_inputs)[0]))
    random.shuffle(index_list)
    train_inputs = tf.gather(train_inputs, index_list)
    train_labels = tf.gather(train_labels, index_list)
    previous_index = 0
    list_losses = []

    for i in range(0, train_inputs.shape[0], model.batch_size):
        if i != 0:
            current_batch = train_inputs[previous_index:i, :, :,
                            :]  # Access the range of values within each batch for inputs
            current_labels = train_labels[previous_index:i]  # Accessing the current batch of labels indexed
            previous_index = i  # Place holder to underlie the indexing of batching

            with tf.GradientTape() as tape:
                logits = model.call(current_batch)  # Get the logits for each model
                loss = model.loss(logits, current_labels)  # get the loss for each batch

            if i % 20 == 0:
                train_acc = model.accuracy(logits, current_labels)
                print("Accuracy on training set after {} training steps: {}".format(i, train_acc))

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            list_losses.append(loss)

            print("Training on batch #: {} loss: {}".format(i / model.batch_size, loss))
    return list_losses


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
    previous_index = 0
    accuracy = []
    for i in range(0, test_inputs.shape[0], model.batch_size):
        if i != 0:
            current_batch = test_inputs[previous_index:i, :, :,
                            :]  # Access the range of values within each batch for inputs
            current_labels = test_labels[previous_index:i]  # Accessing the current batch of labels indexed
            previous_index = i  # Place holder to underlie the indexing of batching
            logits = model.call(current_batch)  # Get the logits for each model
            accuracy.append(model.accuracy(logits, current_labels))

    return np.mean(accuracy)


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list
    field

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """

    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i + 1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs.

    CS1470 students should receive a final accuracy
    on the testing examples for cat and dog of >=70%.

    CS2470 students should receive a final accuracy
    on the testing examples for cat and dog of >=75%.

    :return: None
    '''
    train_path = "/Users/jmuneton/Desktop/Fall2020/CSCI1470/homework/hw2/hw2-cnn-jmuneton-master/data/train"
    test_path = "/Users/jmuneton/Desktop/Fall2020/CSCI1470/homework/hw2/hw2-cnn-jmuneton-master/data/test"
    number_1 = 1
    number_2 = 0
    train_inputs, train_labels = get_data(train_path, number_1, number_2)
    test_inputs, test_labels = get_data(test_path, 1, 0)
    model = Model()
    EPOCH = 15
    # Training data by the number of epochs
    for epoch in range(EPOCH):
        print("*********************** EPOCH NUMBER: {}************************".format(epoch))
        train(model, tf.image.random_flip_left_right(train_inputs), train_labels)

    print("********************* STARTING TESTING: *************************")
    print("FINAL ACCURACY: {}".format(test(model, test_inputs, test_labels)))


if __name__ == '__main__':
    main()
