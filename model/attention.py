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


def attn_model_snippet(model_layer, nb_layers_out, kernel_size, dilation_rate, batch_norm=True, l2_reg=0):
    # Contains the repeating structural part of the model to be used in the creating of the final model

    def add_common_layers(x):
        x = layers.Activation('relu')(x)
        if batch_norm:
            # axis = -1 indicates channel normalization
            x = layers.BatchNormalization(axis=-1, gamma_regularizer=l2(l2_reg))(x)
        return x

    model_layer = layers.Convolution1D(nb_layers_out,
                                       kernel_size=kernel_size,
                                       dilation_rate=dilation_rate,
                                       padding='causal')(model_layer)

    model_layer = add_common_layers(model_layer)
    attention_layer = Attention()(model_layer)

    return model_layer, attention_layer


def create_attn_model(input_shape, learning_rate, l_2_reg=0, batch_norm=True):
    input_layer = Input(shape=(input_shape[0], input_shape[1]))

    x = input_layer
    model_layers = []
    ''' create a list for layers in the format (output, kernel_size, dilation_rate)
     to prevent rewriting the model_snippet function '''

    l_1 = (32, 1, 1)
    model_layers.append(l_1)
    l_2 = (32, 1, 2)
    model_layers.append(l_2)
    l_3 = (32, 1, 4)
    model_layers.append(l_3)
    l_4 = (32, 1, 8)
    model_layers.append(l_4)
    l_5 = (32, 1, 16)
    model_layers.append(l_5)
    l_6 = (32, 2, 32)
    model_layers.append(l_6)

    within_layer_attentions = []
    for l in model_layers:
        x, attn = attn_model_snippet(x, l[0], l[1], l[2], batch_norm, l_2_reg)
        within_layer_attentions.append(attn)

    # across-layer attention
    x = Concatenate()(within_layer_attentions)
    # x has shape (# of within_layer_attentions , timesteps), but Attention takes in (timesteps , # of kernel filters)
    x = layers.Permute((2, 1))(x)
    x = Attention()(x)

    # classification
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(4, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    output_layer = x

    model = Model(input_layer, output_layer)
    model.summary()
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate))
    return model