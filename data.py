# coding: utf-8
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


class MNIST:

    def __init__(self, batch_size=64, data_aug=True):
        self.batch_size = batch_size
        (self.x_train, self.y_train), (self.x_valid, self.y_valid) = tf.keras.datasets.mnist.load_data()
        self.x_train = self.x_train / 127.5 - 1
        self.x_valid = self.x_valid / 127.5 - 1
        self.train_size = len(self.x_train)
        self.valid_size = len(self.x_valid)
        self.img_shape = self.x_train[0].shape
        self.x_train = self.x_train.reshape(self.train_size, self.img_shape[0], self.img_shape[1], 1)
        self.x_valid = self.x_valid.reshape(self.valid_size, self.img_shape[0], self.img_shape[1], 1)
        if data_aug:
            _train_gen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2)
        else:
            _train_gen = ImageDataGenerator()
        _valid_gen = ImageDataGenerator()
        self.train_img_gen = _train_gen.flow(self.x_train, self.y_train, batch_size=self.batch_size)
        self.valid_img_gen = _valid_gen.flow(self.x_valid, self.y_valid, batch_size=self.batch_size)

    @property
    def num_of_batch(self):
        return self.train_size // self.batch_size
