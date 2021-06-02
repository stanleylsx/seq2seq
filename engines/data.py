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
        self.token_file = configs['token_file']
        self.data_path = configs['data_path']
        if not os.path.isfile(self.token_file):
            self.logger.info('vocab files not exist...')
        else:
            self.token2id, self.id2token = self.load_vocab()
            self.vocab_size = len(self.token2id)

    def load_vocab(self):
        if not os.path.isfile(self.token_file):
            self.logger.info('vocab files not exist, building vocab...')
            return self.build_vocab()
        token2id, id2token = {}, {}
        with open(self.token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                token, token_id = row.split('\t')[0], int(row.split('\t')[1])
                token2id[token] = token_id
                id2token[token_id] = token
        self.vocab_size = len(token2id)
        return token2id, id2token

    def build_vocab(self):
        tokens = []

    def tokenize(lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return tensor, lang_tokenizer

    def get_dataset(self):
        dataset = pd.read_csv(self.data_path, encoding='utf-8')
        dataset['origin'] = dataset.origin.apply(preprocess_sentence)
        dataset['target'] = dataset.target.apply(preprocess_sentence)



