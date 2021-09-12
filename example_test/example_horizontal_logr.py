# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 3:03 下午
# @Author  : Alioth
# @File    : example_horizontal_logr.py
# @Email   : thxthx1999@gmail.com
# @Software: PyCharm

import os
import json
import pandas as pd
import numpy as np
import copy
from src.library import OneHotEncoder, ActivateSoftmaxSigmoid, CrossEntropy, Regularization, AccuracyScore

if __name__ == '__main__':

    conf_json = {
        "punishEnum": "L2",
        "tolerate": 0.0001,
        "punish_ratio": 0.5,
        "optimizeEnum": "sgd",
        "batchSize": 1,
        "learn_ratio": 0.0001,
        "local_ep": 5,
        "classificationEnum": "ovr"
    }
    path_project = os.path.abspath('..')
    path_conf = os.path.join(path_project, 'data', 'config' + '.json')
    print(path_conf)
    with open(path_conf, "w") as f:
        json.dump(conf_json, f, indent=4)
    path_train = os.path.join(path_project, 'data', 'train_data_scaled' + '.csv')
    path_test = os.path.join(path_project, 'data', 'test_data_scaled' + '.csv')
    data_train = pd.read_csv(path_train, header=0, index_col=0)
    data_test = pd.read_csv(path_train, header=0, index_col=0)
    n_samples = len(data_train)  # the number of the client's data_train(the number of id), used for averaging
    n_features = len(data_train.columns)

    metrics = {
        '0': {'x1': 1.1, 'x2': 0.6, '1': 1},
        '1': {'x1': 1.3, 'x2': 0.8, '1': 1}
    }

    # one-hot encoding & get data
    data_label = OneHotEncoder(data_train['not.fully.paid'].astype(str), metrics)
    data_features = data_train.drop('not.fully.paid', axis=1)
    data_features.insert(len(data_features.columns), '1', np.ones((n_samples, 1)))  # n*x
    data_features = np.array(data_features)
    test_label = OneHotEncoder(data_test['not.fully.paid'].astype(str), metrics)
    test_features = data_test.drop('not.fully.paid', axis=1)
    test_features.insert(len(test_features.columns), '1', np.ones((n_samples, 1)))  # n*x
    test_features = np.array(test_features)

    parameters = np.zeros((2, n_features))
    parameters[0] = np.random.randn(n_features)
    parameters = parameters.T

    batch_loss = []
    loss = 0
    batchsize = 1000
    for lc_round in range(20):
        batch_num = int(np.ceil(n_samples / batchsize))
        mix_ids = np.random.permutation(n_samples)
        for lc_iter in range(batch_num):
            start = lc_iter * batchsize
            end = (lc_iter + 1) * batchsize
            if end > n_samples:
                end = n_samples
            batch_ids = mix_ids[start: end]
            x = data_label[batch_ids]

            # linear model
            z = np.dot(data_features[batch_ids], parameters)  # n*2 ndarray

            # activation function
            y_predicted = ActivateSoftmaxSigmoid(z)  # y_predicted = torch.sigmoid(z)  # n*2

            # loss function
            criterion, gradients = CrossEntropy(y_predicted, data_label[batch_ids], data_features[batch_ids])

            # re_loss, re_grad = Regularization(parameters.T, 'L2', 0.5)
            #
            # criterion += re_loss  # n*1
            # gradients += re_grad  # n*len(y)*x
            loss = np.sum(criterion) / (end - start)  # 返回的为数值
            grad = np.sum(gradients, axis=0) / (end - start)
            parameters_new = parameters - 0.5 * grad.T  # parameters 纵向 x*len(y)
            parameters_new[:, 1] = 0
            # if np.linalg.norm(parameters_new - parameters) / parameters.size < tolerate:
            #     break
            parameters = copy.deepcopy(parameters_new)
            print(loss)
        z = np.dot(test_features, parameters)  # n*2 ndarray
        # activation function
        y_predicted = ActivateSoftmaxSigmoid(z)  # y_predicted = torch.sigmoid(z)  # n*2
        # test accuracy
        score = AccuracyScore(y_predicted, test_label)
        print(score)
        print('one epoch done')
        batch_loss.append(loss)

    print(batch_loss)
