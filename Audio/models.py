from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from modules import *


slim = tf.contrib.slim

def recurrent_model(net, hidden_units=256, number_of_outputs=3):

    with tf.variable_scope("recurrent"):
      batch_size, seq_length, num_features = net.get_shape().as_list()

      lstm = tf.contrib.rnn.LSTMCell(hidden_units,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)

      #stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2], state_is_tuple=True)


      outputs, states = tf.nn.dynamic_rnn(lstm, net, dtype=tf.float32)

      net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))

      prediction = tf.nn.tanh(slim.layers.linear(net, number_of_outputs))

      return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))



def audio_model2(audio_frames=None, conv_filters=40):

    with tf.variable_scope("audio_model"):
      #print(audio_frames)
      batch_size, seq_length, num_features = audio_frames.get_shape().as_list()
      audio_input = tf.reshape(audio_frames, [batch_size ,  num_features * seq_length, 1])

      #print(net)
      net = tf.layers.conv1d(audio_input,50,8,padding = 'same', activation=tf.nn.relu)
      net = tf.layers.max_pooling1d(net,10,10)
      net = slim.dropout(net,0.5)


      #print(net)

      net = tf.layers.conv1d(net,125,6,padding = 'same', activation =tf.nn.relu)
      net = tf.layers.max_pooling1d(net, 5, 5)
      net = slim.dropout(net,0.5)

      #print(net)

      net = tf.layers.conv1d(net,250,6,padding = 'same', activation =tf.nn.relu)
      net = tf.layers.max_pooling1d(net,5,5)
      net = slim.dropout(net,0.5)

      #print(net)

      net = tf.reshape(net,[batch_size,seq_length,-1])
      print(net)

    return net

def get_model(name):

    name_to_fun = {'audio_model2': audio_model2}

    if name in name_to_fun:
        model = name_to_fun[name]
    else:
        raise ValueError('Requested name [{}] not a valid model'.format(name))

    def wrapper(*args, **kwargs):
        return recurrent_model(model(*args), **kwargs)

    return wrapper
