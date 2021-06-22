# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py 
# @Software: PyCharm


# [train, interactive_predict]
mode = 'interactive_predict'

CUDA_VISIBLE_DEVICES = 0
# int, -1:CPU, [0,]:GPU
# coincides with tf.CUDA_VISIBLE_DEVICES

configs = {
    # 任务选择
    'task': 'translation',
    'data_path': 'datasets/translation/spa-eng/spa.csv',
    'token_dir': 'datasets/translation/spa-eng/',
    'batch_size': 64,
    # 每print_per_batch打印
    'print_per_batch': 100,
    # 向量维度
    'embedding_dim': 256,
    # 是否双向
    'bidirectional': False,
    # 编码器隐藏层维度
    'encoder_hidden_dim': 512,
    # 解码器隐藏层维度
    'decoder_hidden_dim': 512,
    # RNN Type:
    # 可选:lstm, gru
    'rnn_type': 'gru',
    # 训练epoch
    'epoch': 30,
    # 最多保存max_to_keep个模型
    'max_to_keep': 1,
    # 模型保存的文件夹
    'checkpoints_dir': 'model/seq2seq',
    # 模型保存的名字
    'checkpoint_name': 'seq2seq',
}
