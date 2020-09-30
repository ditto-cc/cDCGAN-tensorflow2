# coding: utf-8
import glob
import math
import os

import imageio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tqdm

import utils
from utils import path
from model import generator, discriminator


class CDCGAN:
    def __init__(self):

        self.ckpt_path = utils.CKPT_PATH
        self.gen = generator()
        self.dis = discriminator()

        self.gen.summary()
        self.dis.summary()

        self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.dis_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.init_epoch = tf.Variable(0)

        self.summary_writer = tf.summary.create_file_writer(utils.SUMMARY_PATH)
        self.ckpt = tf.train.Checkpoint(
            init_epoch=self.init_epoch,
            gen=self.gen,
            dis=self.dis,
            gen_optimizer=self.gen_optimizer,
            dis_optimizer=self.dis_optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=self.ckpt_path, max_to_keep=3)
        self.gen_img_path = path.join(utils.OUTPUT_PATH, 'mnist')
        if not path.exists(self.gen_img_path):
            os.mkdir(self.gen_img_path)

        if not path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)

    def load_latest_checkpoint(self):
        """加载最新检查点"""
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self.gen = self.ckpt.gen
            self.dis = self.ckpt.dis
            self.gen_optimizer = self.ckpt.gen_optimizer
            self.dis_optimizer = self.ckpt.dis_optimizer
            self.init_epoch = self.ckpt.init_epoch
            print('Latest Checkpoint Restore.')

    def generate_and_save_images(self, epoch, test_input, num_to_gen=100):
        """生成测试图片并保存"""
        n = int(math.sqrt(num_to_gen))

        labels = np.arange(0, num_to_gen)
        labels = labels % 10
        labels = labels.reshape((num_to_gen, 1, 1, 1))
        labels = tf.keras.utils.to_categorical(labels)
        predictions = self.gen([test_input, labels], training=False)

        plt.title(f'Epoch {epoch}')
        plt.figure(figsize=(n, n))
        for i in range(predictions.shape[0]):
            plt.subplot(n, n, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        save_filename = path.join(self.gen_img_path, 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.savefig(save_filename)
        plt.close()

    def gen_gif(self):
        """生成GIF图记录测试图片的变化过程"""
        anim_file = path.join(self.gen_img_path, 'mnist_gen_process.gif')
        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(path.join(self.gen_img_path, 'image*.png'))
            filenames = sorted(filenames)
            last = -1
            for i, filename in enumerate(filenames):
                frame = 2 * (i ** 0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

    def train(self, train_dataset, epochs=50):
        """训练过程"""

        def gen_loss(fake):
            """LS-GAN"""
            return tf.reduce_mean(tf.square(fake - 1.0))

        def dis_loss(real, fake):
            """LS-GAN"""
            return tf.reduce_mean(tf.square(fake)) + tf.reduce_mean(tf.square(real - 1.0))

        # 记录训练过程中的损失值
        gen_loss_val = tf.keras.metrics.Mean(name='gen_loss')
        dis_loss_val = tf.keras.metrics.Mean(name='dis_loss')

        # 单步训练
        @tf.function
        def train_step(train_imgs, train_gen_labels, train_dis_labels):
            noise = tf.random.normal([len(train_labels), 1, 1, 100])

            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
                fake_imgs = self.gen([noise, train_gen_labels], training=True)
                real_dis_output = self.dis([train_imgs, train_dis_labels], training=True)
                fake_dis_output = self.dis([fake_imgs, train_dis_labels], training=True)

                disc_loss_val = dis_loss(real_dis_output, fake_dis_output)
                gene_loss_val = gen_loss(fake_dis_output)

            gen_loss_val(gene_loss_val)
            dis_loss_val(disc_loss_val)

            gen_grad = g_tape.gradient(gene_loss_val, self.gen.trainable_variables)
            self.gen_optimizer.apply_gradients(zip(gen_grad, self.gen.trainable_variables))

            dis_grad = d_tape.gradient(disc_loss_val, self.dis.trainable_variables)
            self.dis_optimizer.apply_gradients(zip(dis_grad, self.dis.trainable_variables))

        # 加载检查点
        self.load_latest_checkpoint()

        # 测试图片数量
        num_img_to_gen = 100
        # 测试初始随机噪声
        test_input = tf.random.normal([num_img_to_gen, 1, 1, 100])

        train_gen = train_dataset.train_img_gen
        # 训练循环
        for epoch in range(self.init_epoch.read_value(), epochs):
            desc = f'Epoch {epoch + 1}/{epochs}, Step'
            # 每个训练循环，训练一遍数据集的图片，Batch默认256张
            for _ in tqdm.trange(train_dataset.num_of_batch, desc=desc, total=train_dataset.num_of_batch):
                train_imgs, train_labels = next(train_gen)
                train_labels = tf.keras.utils.to_categorical(train_labels)
                train_gen_labels = train_labels.reshape([len(train_labels), 1, 1, 10])
                train_dis_labels = tf.ones((len(train_labels), 28, 28, 10)) * train_gen_labels
                train_step(train_imgs, train_gen_labels, train_dis_labels)

            # 记录当前训练步数
            self.init_epoch.assign_add(1)

            # 保存检查点
            if (epoch + 1) % 10 == 0:
                self.ckpt_manager.save()
                print('Save Checkpoint')

            # 记录损失值，写入summary
            g_loss_val = gen_loss_val.result()
            d_loss_val = dis_loss_val.result()
            print(f'gen_loss: {g_loss_val}, '
                  f'dis_loss: {d_loss_val}')
            with self.summary_writer.as_default():
                tf.summary.scalar('gen_loss', g_loss_val, step=epoch, description='generator loss')
                tf.summary.scalar('dis_loss', d_loss_val, step=epoch, description='discriminator loss')
            gen_loss_val.reset_states()
            dis_loss_val.reset_states()

            # 生成保存图像
            self.generate_and_save_images(epoch + 1, test_input, num_img_to_gen)
