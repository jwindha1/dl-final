from __future__ import absolute_import
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import numpy as np
import random
import math
from attention import Attention

WINDOW_SIZE = 20

class TCNModel(tf.keras.Model):
    def __init__(self, learning_rate=0.005):
        """
        This model class will contain the architecture for our TCN
        """
        super(TCNModel, self).__init__()

        self.batch_size = 64
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.conv1 = tf.keras.layers.Conv1D(filters=32, strides=1, padding="causal", kernel_size=4, dilation_rate=1, input_shape=(WINDOW_SIZE, 1))
        self.att1 = Attention(1)
        self.conv2 = tf.keras.layers.Conv1D(filters=16, strides=1, padding="causal", kernel_size=4, dilation_rate=2)
        self.att2 = Attention(1)
        self.conv3 = tf.keras.layers.Conv1D(filters=8, strides=1, padding="causal", kernel_size=4, dilation_rate=4)
        self.att3 = Attention(1)
        self.conv4 = tf.keras.layers.Conv1D(filters=4, strides=1, padding="causal", kernel_size=4, dilation_rate=8)
        self.att4 = Attention(1)
        self.conv5 = tf.keras.layers.Conv1D(filters=2, strides=1, padding="causal", kernel_size=4, dilation_rate=16)
        self.att5 = Attention(1)
        self.conv6 = tf.keras.layers.Conv1D(filters=2, strides=1, padding="causal", kernel_size=4, dilation_rate=32)
        self.att6 = Attention(1)
        self.across_attn = Attention(2)
        self.dense = tf.keras.layers.Dense(units=2)


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 20)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        inputs = tf.expand_dims(inputs, axis=-1)
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
        # print(f'at each layer:')
        # print(f'c1: {c1.shape}, c2: {c2.shape}, c3: {c3.shape}, c4: {c4.shape}, c5: {c5.shape}, c6: {c6.shape},')
        # print(f'gamma: {gamma1.shape}, gamma2: {gamma2.shape}, gamma3: {gamma3.shape}, gamma4: {gamma4.shape}, gamma5: {gamma5.shape}, gamma6: {gamma6.shape},')
        within_attention  = tf.concat([gamma1, gamma2, gamma3, gamma4, gamma5, gamma6], axis=1)
        final_gamma = self.across_attn.call(within_attention)
        # print(f'final: within attention {within_attention.shape}, gamma {final_gamma.shape}')
        output = self.dense(final_gamma)
        return output

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
