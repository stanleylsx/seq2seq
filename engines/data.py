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
from sklearn.model_selection import train_test_split
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

        self.origin_vocab_size = 0
        self.target_vocab_size = 0

        self.origin_max_len = 0
        self.target_max_len = 0

        self.origin_token2id, self.origin_id2token = {}, {}
        self.target_token2id, self.target_id2token = {}, {}

        if not os.path.isfile(self.token_dir + '/origin_token2id'):
            self.logger.info('vocab files not exist...')
        else:
            self.origin_token2id, self.origin_id2token, self.origin_vocab_size, self.origin_max_len = \
                self.load_vocab('origin')
            self.target_token2id, self.target_id2token, self.target_vocab_size, self.target_max_len = \
                self.load_vocab('target')

    def load_vocab(self, name):
        token2id, id2token = {}, {}
        with open(self.token_dir + '/' + name + '_token2id', 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                token, token_id = row.split('\t')[0], int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token
        vocab_size = len(token2id)
        # 加载语料最大长度
        with open(self.token_dir + '/' + name + '_max_lens', 'r', encoding='utf-8') as file:
            max_lens = file.read()
        return token2id, id2token, vocab_size, max_lens

    def tokenize(self, sentences, name):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        tokenizer.fit_on_texts(sentences)
        token2id = tokenizer.word_index
        token2id[self.PADDING] = 0
        id2token = tokenizer.index_word
        id2token[0] = self.PADDING
        vocab_size = len(id2token)
        # 保存词表及标签表
        with open(self.token_dir + '/' + name + '_token2id', 'w', encoding='utf-8') as outfile:
            for token, token_id in tqdm(token2id.items()):
                outfile.write(token + '\t' + str(token_id) + '\n')
        tensor = tokenizer.texts_to_sequences(sentences)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        max_len = int(tf.shape(tensor)[-1])
        # 保存语料最大长度
        with open(self.token_dir + '/' + name + '_max_lens', 'w', encoding='utf-8') as outfile:
            outfile.write(str(max_len))
        return tensor, vocab_size, token2id, id2token, max_len

    def get_dataset(self):
        dataset = pd.read_csv(self.data_path, encoding='utf-8')[:3000]
        dataset['origin'] = dataset.origin.apply(preprocess_sentence)
        origin_tensor, self.origin_vocab_size, self.origin_token2id, self.origin_id2token, self.origin_max_len = \
            self.tokenize(dataset['origin'], 'origin')
        dataset['target'] = dataset.target.apply(preprocess_sentence)
        target_tensor, self.target_vocab_size, self.target_token2id, self.target_id2token, self.target_max_len = \
            self.tokenize(dataset['target'], 'target')
        origin_tensor_train, origin_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
            origin_tensor, target_tensor, test_size=0.2)
        dataset = tf.data.Dataset.from_tensor_slices((origin_tensor_train, target_tensor_train))
        return dataset
