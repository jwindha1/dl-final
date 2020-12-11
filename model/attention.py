import numpy as np
import argparse
import tensorflow as tf
from datetime import datetime

# heavy keras imports
from keras.layers import Dense, Flatten, Input, Activation, Add, Concatenate
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.callbacks import TensorBoard
from keras import layers
from keras.regularizers import l2
from keras import Model
from keras import backend as K


class Attention(layers.Layer):
    '''
    Custom Attention Layer
    '''


    def __init__(self, output_size):
        super(Attention, self).__init__()
        self.dense = tf.keras.layers.Dense(units =output_size)

    def call(self, input):
        '''
        The mathematical equation that we are trying to implement here is as follows:

        alpha = softmax(tanh(input*W)) , where alpha is the attention weight with shape = (timesteps, 1)
        gamma = Relu(transpose(input)*alpha)

        :param input: output of the convolution activation layer, shape = (timesteps, number of kernel filters)
        :return: within layer attention output (gamma), shape = (number of kernel filters, 1)
        '''
        
        input_for_alpha = self.dense(input)
        alpha = tf.nn.softmax(tf.nn.tanh(input_for_alpha))
        gamma = tf.nn.relu(tf.matmul(input, alpha, transpose_a=True))
        # print(f'attn call input: {input.shape}, input_for_alpha: {input_for_alpha.shape}, alpha: {alpha.shape}, gamma: {gamma.shape}')
        return tf.squeeze(gamma)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], 1)


