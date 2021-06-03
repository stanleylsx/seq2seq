# -*- coding: utf-8 -*-
# @Time : 2021/6/1 18:39 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py 
# @Software: PyCharm
import tensorflow as tf
import pandas as pd
import re
import os
from tqdm import tqdm
from config import configs
from engines.utils.translation_utils import preprocess_sentence


class TranslationDataManager:

    def __init__(self, logger):
        self.logger = logger
        self.token_dir = configs['token_dir']
        self.data_path = configs['data_path']

        self.PADDING = '[pad]'
        self.UNKNOWN = '[unk]'

        # if not os.path.isdir(self.token_dir):
        #     self.logger.info('vocab files not exist...')
        # else:
        #     self.vocab_size = len(self.token2id)

    # def load_vocab(self):
    #     token2id, id2token = {}, {}
    #     with open(self.token_file, 'r', encoding='utf-8') as infile:
    #         for row in infile:
    #             row = row.strip()
    #             token, token_id = row.split('\t')[0], int(row.split('\t')[1])
    #             token2id[token] = token_id
    #             id2token[token_id] = token
    #     self.vocab_size = len(token2id)
    #     return token2id, id2token

    def tokenize(self, sentences, name):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(sentences)
        token2id = tokenizer.word_index
        token2id[self.PADDING] = 0
        id2token = tokenizer.index_word
        id2token[0] = self.PADDING
        # 保存词表及标签表
        with open(self.token_dir + '/' + name + '_token2id', 'w', encoding='utf-8') as outfile:
            for token, token_id in token2id.items():
                outfile.write(token + '\t' + str(token_id) + '\n')
        tensor = tokenizer.texts_to_sequences(sentences)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return tensor

    def get_dataset(self):
        dataset = pd.read_csv(self.data_path, encoding='utf-8')
        dataset['origin'] = dataset.origin.apply(preprocess_sentence)
        origin_tensor = self.tokenize(dataset['origin'], 'origin')
        dataset['target'] = dataset.target.apply(preprocess_sentence)
        target_tensor = self.tokenize(dataset['target'], 'target')
        print()



