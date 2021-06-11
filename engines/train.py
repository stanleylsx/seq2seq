# -*- coding: utf-8 -*-
# @Time : 2021/6/1 18:38 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py 
# @Software: PyCharm
import tensorflow as tf
import time
from engines.seq2seq import Encoder, Decoder
from config import configs
from tqdm import tqdm


def train(data_manager, logger):
    datasets = data_manager.get_dataset()
    epoch = configs['epoch']
    batch_size = configs['batch_size']
    print_per_batch = configs['print_per_batch']

    max_to_keep = configs['max_to_keep']
    checkpoints_dir = configs['checkpoints_dir']
    checkpoint_name = configs['checkpoint_name']

    target_token2id = data_manager.target_token2id

    origin_vocab_size = data_manager.origin_vocab_size
    target_vocab_size = data_manager.target_vocab_size
    encoder = Encoder(origin_vocab_size)
    decoder = Decoder(target_vocab_size)

    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoints_dir, checkpoint_name=checkpoint_name, max_to_keep=max_to_keep)

    logger.info(('+' * 20) + 'training starting' + ('+' * 20))
    very_start_time = time.time()
    for i in range(epoch):
        start_time = time.time()
        total_loss = 0.0
        logger.info('Epoch:{}/{}'.format(i + 1, epoch))
        steps = int(tf.math.ceil(len(datasets) / batch_size))
        for step, batch in tqdm(datasets.shuffle(len(datasets)).batch(batch_size).enumerate()):
            origin_batch_train, target_batch_train = batch
            with tf.GradientTape() as tape:
                encoder_output, encoder_hidden = encoder(origin_batch_train)
                decoder_hidden = encoder_hidden
                decoder_input = tf.expand_dims([target_token2id['[start]']] * batch_size, 1)
                step_loss = 0.0
                indies = len(target_batch_train[1].numpy())
                for t in range(1, indies):
                    # 将编码器输出 （enc_output） 传送至解码器
                    predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
                    y_true = target_batch_train[:, t]
                    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
                    loss_vec = tf.losses.sparse_categorical_crossentropy(y_true=y_true, y_pred=predictions)
                    mask = tf.cast(mask, dtype=loss_vec.dtype)
                    loss_vec = loss_vec * mask
                    loss = tf.reduce_mean(loss_vec)
                    step_loss = step_loss + loss
                    # using teacher forcing
                    decoder_input = tf.expand_dims(y_true, 1)
                batch_loss = (step_loss / int(target_batch_train.shape[1]))
                total_loss += batch_loss
            variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(batch_loss, variables)
            # 反向传播，自动微分计算
            optimizer.apply_gradients(zip(gradients, variables))

            if step % print_per_batch == 0 and step != 0:
                logger.info('Epoch {} Step {} Loss {:.4f}'.format(i + 1, step, batch_loss))

        time_span = (time.time() - start_time) / 60
        logger('Epoch {} Loss {:.4f}'.format(i + 1, total_loss / steps))
        logger.info('Time consumption:%.2f(min)' % time_span)

        # 每两个epoch保存一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint_manager.save()
            logger.info('Saved the new model')
    logger.info('Total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))



