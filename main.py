# -*- coding: utf-8 -*-
# @Time : 2021/6/1 17:36 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py 
# @Software: PyCharm
from engines.data import *
from engines.utils.logger import get_logger
from engines.train import train
from config import mode, configs, CUDA_VISIBLE_DEVICES
import os
import json


if __name__ == '__main__':
    logger = get_logger('./logs')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_VISIBLE_DEVICES)
    if mode == 'train':
        logger.info(json.dumps(configs, indent=2))
        if configs['task'] == 'translation':
            data_manage = TranslationDataManager(logger)
        else:
            data_manage = TranslationDataManager(logger)
        logger.info('mode: train')
        train(data_manage, logger)
