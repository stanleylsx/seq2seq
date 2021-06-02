# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py 
# @Software: PyCharm


# [train, interactive_predict]
mode = 'train'

CUDA_VISIBLE_DEVICES = 0
# int, -1:CPU, [0,]:GPU
# coincides with tf.CUDA_VISIBLE_DEVICES

configs = {
    # 任务选择
    'task': 'translation',
    'data_path': 'datasets/translation/spa-eng/spa.csv',
    'token_file': 'datasets/translation/spa-eng/token2id',

}
