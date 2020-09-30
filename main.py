# coding: utf-8
from data import MNIST
from train import CDCGAN

if __name__ == '__main__':
    cdcgan = CDCGAN()
    mnist = MNIST()
    cdcgan.train(mnist)
    pass
