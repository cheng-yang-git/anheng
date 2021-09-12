# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/5 11:09 上午
# @Author  : Alioth
# @File    : library.py
# @Email   : thxthx1999@gmail.com
# @Software: PyCharm

import numpy as np
from numpy import argmax
from sklearn import preprocessing


def OneHotEncoder(labels, metrics):

    Y = list(metrics.keys())

    # integer encode
    le = preprocessing.LabelEncoder()
    integer_encoded = le.fit_transform(Y)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # binary encode
    enc = preprocessing.OneHotEncoder()  # don't drop='first'
    onehot_encoded = enc.fit_transform(integer_encoded).toarray()

    onehot_labels = np.ones((len(labels), onehot_encoded.shape[1]))
    for i, y in enumerate(labels):  # Series
        onehot_labels[i] = onehot_encoded[Y.index(y)]

    return onehot_labels


def ActivateSoftmaxSigmoid(z):
    """稳定的Softmax，防止溢出"""
    # sigmoid = 1 / (1 + np.exp(-z))  # n*1 return size(z) sigmoid = e^z / (1 + e^z)
    shiftz = z - np.max(z)  # [1000, 2000, 3000] -> [0., 0., 1.]
    softmaxsigmoid = np.exp(shiftz)/np.sum(np.exp(shiftz), axis=1, keepdims=True)  # n*len(y) return size(z)
    # keepdims=True 保持二维特性
    return softmaxsigmoid


def CrossEntropy(y_predicted, labels, x_data):
    y = np.array(labels)  # n*len(y)
    x = np.array(x_data)
    loss = -np.sum(y * np.log(y_predicted), axis=1).reshape(-1, 1)  # n*1
    grad = np.ones((len(x), y_predicted.shape[1], x.shape[1]))  # n*len(y)*x
    temp = y_predicted - y
    for i in range(len(x)):
        grad[i] = temp[i].reshape(-1, 1) * x_data[i]
    return loss, grad


def Regularization(theta, punish, punish_ratio):  # theta ndarray len(y)*x
    if punish == 'L1':
        regularization_loss = np.sum(np.abs(theta)) * punish_ratio
        regularization_partial = np.sign(theta) * punish_ratio
    elif punish == 'L2':
        regularization_loss = np.sum(theta**2) * punish_ratio / 2
        regularization_partial = theta * punish_ratio
    else:
        regularization_loss = 0
        regularization_partial = 0
    return regularization_loss, regularization_partial


def AccuracyScore(y_predicted, labels):
    idx = np.argmax(y_predicted, axis=1)  # y_predicted n*len(y)
    y_p = (idx[:, None] == np.arange(y_predicted.shape[1])).astype(int)
    y = np.array(labels)  # n*len(y)
    c = (y_p == y).all(axis=1)
    score = c.sum() / len(c)
    return score


def OneHotEncoder_Python():
    # define input string
    data = 'hello world'
    print(data)
    # define universe of possible input values
    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    # integer encode input data
    integer_encoded = [char_to_int[char] for char in data]
    print(integer_encoded)
    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    print(onehot_encoded)
    # invert encoding
    inverted = int_to_char[argmax(onehot_encoded[0])]
    print(inverted)
