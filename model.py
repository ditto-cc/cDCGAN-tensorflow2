# coding: utf-8


import tensorflow as tf
from tensorflow.keras import layers

import utils


def generator(name='gen'):
    k_init = tf.keras.initializers.he_normal()
    b_init = tf.keras.initializers.he_normal()

    # 随机噪声
    inputs = layers.Input(shape=(1, 1, 100), name='noise')
    # 标注值，One-Hot编码
    labels = layers.Input(shape=(1, 1, 10), name='label')

    # 1, 1, 110
    x = layers.concatenate([inputs, labels])

    # 7, 7, 256
    x = layers.Conv2DTranspose(filters=256, kernel_size=7,
                               strides=1, padding='valid',
                               kernel_initializer=k_init,
                               bias_initializer=b_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 14, 14, 128
    x = layers.Conv2DTranspose(filters=128, kernel_size=5,
                               strides=2, padding='same',
                               kernel_initializer=k_init,
                               bias_initializer=b_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 28, 28, 1
    x = layers.Conv2DTranspose(filters=1, kernel_size=5,
                               strides=2, padding='same',
                               kernel_initializer=k_init,
                               bias_initializer=b_init)(x)
    x = layers.Activation(tf.nn.tanh)(x)

    return tf.keras.Model(inputs=[inputs, labels], outputs=x, name=name)


def discriminator(name='dis'):
    k_init = tf.keras.initializers.he_normal()
    b_init = tf.keras.initializers.he_normal()

    inputs = layers.Input(shape=(28, 28, 1), name='image')
    labels = layers.Input(shape=(28, 28, 10), name='label')

    x = layers.concatenate([inputs, labels])
    x = layers.Conv2D(filters=128, kernel_size=5,
                      strides=2, padding='same',
                      kernel_initializer=k_init,
                      bias_initializer=b_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters=256, kernel_size=5,
                      strides=2, padding='same',
                      kernel_initializer=k_init,
                      bias_initializer=b_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters=1, kernel_size=7,
                      strides=1, padding='same',
                      kernel_initializer=k_init,
                      bias_initializer=b_init)(x)

    return tf.keras.Model(inputs=[inputs, labels], outputs=x, name=name)


if __name__ == '__main__':
    summary_path = utils.SUMMARY_PATH
    tf.keras.utils.plot_model(generator(), to_file=utils.join(summary_path, 'gen_model.png'), show_shapes=True, dpi=64)
    tf.keras.utils.plot_model(discriminator(), to_file=utils.join(summary_path, 'dis_model.png'), show_shapes=True, dpi=64)
