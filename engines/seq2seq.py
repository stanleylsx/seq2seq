# -*- coding: utf-8 -*-
# @Time : 2021/6/1 18:22 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : seq2seq.py 
# @Software: PyCharm
from abc import ABC
import tensorflow as tf
from config import configs


class Encoder(tf.keras.Model, ABC):
    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_dim = configs['encoder_hidden_dim']
        self.embedding_dim = configs['embedding_dim']
        self.embedding = tf.keras.layers.Embedding(vocab_size, self.embedding_dim)
        if configs['rnn_type'] == 'gru':
            self.rnn = tf.keras.layers.GRU(self.hidden_dim,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        elif configs['rnn_type'] == 'lstm':
            self.rnn = tf.keras.layers.LSTM(self.hidden_dim,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')

    @tf.function
    def call(self, x):
        x = self.embedding(x)
        output, state = self.rnn(x)
        return output, state


class SequenceToSequence(tf.keras.Model, ABC):
    def __init__(self, origin_vocab_size, target_vocab_size):
        super(SequenceToSequence, self).__init__()
        self.encoder = Encoder(origin_vocab_size)

    @tf.function
    def call(self, origin_input, target_input):
        encoder_output, state = self.encoder(origin_input)

