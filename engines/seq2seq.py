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

    # @tf.function
    def call(self, x):
        x = self.embedding(x)
        output, state = self.rnn(x)
        return output, state


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # @tf.function
    def call(self, query, values):
        # 隐藏层的形状 == （批大小，隐藏层大小）
        # hidden_with_time_axis 的形状 == （批大小，1，隐藏层大小）
        # 这样做是为了执行加法以计算分数
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 分数的形状 == （批大小，最大长度，1）
        # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
        # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
        attention_weights = tf.nn.softmax(score, axis=1)

        # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model, ABC):
    def __init__(self, vocab_size):
        super(Decoder, self).__init__()
        self.hidden_dim = configs['decoder_hidden_dim']
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
        self.fc = tf.keras.layers.Dense(vocab_size)
        # 用于注意力
        self.attention = BahdanauAttention(self.hidden_dim)

    # @tf.function
    def call(self, x, hidden, encoder_output):
        # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
        context_vector, attention_weights = self.attention(hidden, encoder_output)
        # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
        x = self.embedding(x)
        # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # 将合并后的向量传送到RNN
        output, state = self.rnn(x)
        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))
        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)
        return x, state, attention_weights
