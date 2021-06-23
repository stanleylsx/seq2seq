# -*- coding: utf-8 -*-
# @Time : 2021/6/1 18:38 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py 
# @Software: PyCharm
import tensorflow as tf
from config import configs
from engines.seq2seq import Encoder, Decoder
from engines.utils.translation_utils import preprocess_sentence


class Predictor:
    def __init__(self, data_manager, logger):
        self.logger = logger
        self.checkpoints_dir = configs['checkpoints_dir']
        self.checkpoint_name = configs['checkpoint_name']

        self.origin_token2id = data_manager.origin_token2id
        self.target_token2id = data_manager.target_token2id
        self.origin_id2token = data_manager.origin_id2token
        self.target_id2token = data_manager.target_id2token

        self.origin_max_len = data_manager.origin_max_len
        self.target_max_len = data_manager.target_max_len

        origin_vocab_size = data_manager.origin_vocab_size
        target_vocab_size = data_manager.target_vocab_size

        logger.info('loading model parameter')

        self.encoder = Encoder(origin_vocab_size)
        self.decoder = Decoder(target_vocab_size)
        # 实例化Checkpoint，设置恢复对象为新建立的模型
        checkpoint = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
        # 从文件恢复模型参数
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoints_dir))
        logger.info('loading model successfully')

    def translate(self, sentence):
        origin_lang_type = configs['origin_lang_type']
        sentence = preprocess_sentence(sentence, origin_lang_type)
        inputs = [[self.origin_token2id[i] for i in sentence.split(' ')]]
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=int(self.origin_max_len), padding='post')
        inputs = tf.convert_to_tensor(inputs)

        encoder_output, encoder_hidden = self.encoder(inputs)
        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.target_token2id['[start]']], 0)

        result = ''

        for t in range(int(self.target_max_len)):
            predictions, decoder_hidden, attention_weights = self.decoder(
                decoder_input, decoder_hidden, encoder_output)
            predicted_id = tf.argmax(predictions[0]).numpy()
            if self.target_id2token[predicted_id] == '[end]':
                return result, sentence
            result += self.target_id2token[predicted_id] + ' '

            # the predicted ID is fed back into the model
            decoder_input = tf.expand_dims([predicted_id], 0)
        return result, sentence







