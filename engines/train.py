# -*- coding: utf-8 -*-
# @Time : 2021/6/1 18:38 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py 
# @Software: PyCharm
import tensorflow as tf
import time
from engines.seq2seq import SequenceToSequence
from config import configs
from tqdm import tqdm


def train(data_manager, logger):
    datasets = data_manager.get_dataset()
    epoch = configs['epoch']
    batch_size = configs['batch_size']

    origin_vocab_size = data_manager.origin_vocab_size
    target_vocab_size = data_manager.target_vocab_size
    seq2seq = SequenceToSequence(origin_vocab_size, target_vocab_size)

    logger.info(('+' * 20) + 'training starting' + ('+' * 20))
    for i in range(epoch):
        start_time = time.time()
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for step, batch in tqdm(datasets.shuffle(len(datasets)).batch(batch_size).enumerate()):
            origin_batch_train, target_batch_train = batch
            with tf.GradientTape() as tape:
                seq2seq(origin_batch_train, target_batch_train)

