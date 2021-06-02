# -*- coding: utf-8 -*-
# @Time : 2021/6/1 18:38 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py 
# @Software: PyCharm
import tensorflow as tf


def train(data_manager, logger):
    data_manager.get_dataset()
