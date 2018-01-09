# -*- coding: utf-8 -*-
# @Time    : 2018/1/9 16:14
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : dataload.py
# @Software: PyCharm

from keras.datasets import cifar10
import tensorflow as tf

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cifar10.load_data()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = tf.one_hot(Y_train_orig, 10,axis=0)
Y_test = tf.one_hot(Y_test_orig, 10,axis=0)

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


def dataload():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cifar10.load_data()

    # Normalize image vectors
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Convert training and test labels to one hot matrices
    Y_train = tf.one_hot(Y_train_orig, 10, axis=0)
    Y_test = tf.one_hot(Y_test_orig, 10, axis=0)

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    return X_train, Y_train, X_test, Y_test