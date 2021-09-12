# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/2 4:27 下午
# @Author  : Alioth
# @File    : horizontal_logistic_regression.py
# @Email   : thxthx1999@gmail.com
# @Software: PyCharm


import os
import json
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy
from library import OneHotEncoder, ActivateSoftmaxSigmoid, CrossEntropy, Regularization, AccuracyScore


# 在Java中，通过Runtime调用Python脚本时，如果要实时输出脚本的print信息，需要加上参数“-u”。否则Python默认会有输出缓存。python -h有说明
# cmd.add("python");
# cmd.add("-u"); //!!!==加上参数u让脚本实时输出==!!!
# cmd.add("script/python/test.py");


def DeliveryConfiguration(processHistoryId, jobHistoryId, c_json):

    jobhistoryid = jobHistoryId

    config_json = c_json
    path_config = os.path.join('root', "federal", str(processHistoryId), 'config' + '.json')
    with open(path_config, "w") as f:
        json.dump(config_json, f, indent=4)

    return jobhistoryid


def InitialLocalTrainLogR(processHistoryId, jobHistoryId, globalRound, metrics, trainFilePath):

    jobhistoryid = jobHistoryId
    global_round = globalRound
    metrics_updated = copy.deepcopy(metrics)

    path_callbackurl = os.path.join('root', "federal", str(processHistoryId), 'callbackurl' + '.json')
    with open(path_callbackurl, "r") as f:
        json_callbackurl = json.load(f)  # _io.TextIOWrapper
    callBackUrl = json_callbackurl['callBackUrl']

    path_config = os.path.join('root', "federal", str(processHistoryId), 'config' + '.json')
    with open(path_config, "r") as f:
        json_conf = json.load(f, indent=4)
    punishEnum = json_conf['punishEnum']
    punish_ratio = json_conf['punish_ratio']  # weight_decay
    optimizeEnum = json_conf['optimizeEnum']
    batchSize = json_conf['batchSize']
    learn_ratio = json_conf['learn_ratio']
    local_ep = json_conf['local_ep']
    classificationEnum = json_conf['classificationEnum']  # ovr

    path_train = os.path.join('root', "federal", str(processHistoryId), 'train_data' + '.csv')
    if not os.path.samefile(path_train, trainFilePath):
        return
    data_train = pd.read_csv(path_train, header=0, index_col=0)
    n_samples = len(data_train)  # the number of the client's data_train(the number of id), used for averaging
    n_labels = len(metrics)  # the number of multi_classes to determine loss function: sigmod or softmax
    n_features = len(data_train.columns)  # the number of features to determine the number of parameters + bias
    # n_features = len(list(metrics['x1'].keys()))  # Python 3x0

    # one-hot encoding & get data
    data_label = OneHotEncoder(data_train['y'].astype(str), metrics)  # data_train['y']为Series可enumerate遍历 return numpy
    data_features = data_train.drop('y', axis=1)
    data_features.insert(len(data_features.columns), '1', np.ones((n_samples, 1)))  # n*x
    data_features = np.array(data_features)

    if not classificationEnum == 'ovr':
        return

    if n_labels == 2:  # binary classification
        # 默认 bias 放在metrics的最后 w0*x0+ ... + w_(n_features-1)*x_(n_features-1) + b*x_(n_features)
        # updated weights and biases in map<map<>>
        parameters = []
        for y_i, x_j in metrics.items():
            for j, k in x_j.items():
                parameters.append(k)
        parameters = np.array(parameters)  # 横向 1*x
        parameters = parameters.reshape(-1, 1)  # 纵向 x*1
        parameters = np.insert(parameters, 1, np.zeros(len(parameters)), axis=1)
        # insert position = -1, is second to last
    elif n_labels > 2:  # multi-classification
        parameters = [[] for _ in range(n_labels)]
        i = 0
        for y_i, x_j in metrics.items():
            parameters[i] = []
            for j, k in x_j.items():
                parameters[i].append(k)
            i += 1
        parameters = np.array(parameters)  # 横向 len(y)*x
        parameters = parameters.T  # 纵向 x*len(y) or parameters.T reshape(-1, n_labels)
    else:
        return

    if optimizeEnum == 'sgd':
        for lc_round in range(local_ep):
            batch_loss = []
            batch_num = int(np.ceil(n_samples / batchSize))
            mix_ids = np.random.permutation(n_samples)
            for lc_iter in range(batch_num):
                start = lc_iter * batchSize
                end = (lc_iter + 1) * batchSize
                if end > n_samples:
                    end = n_samples
                batch_ids = mix_ids[start: end]

                # linear model
                z = np.dot(data_features[batch_ids], parameters)  # n*2 ndarray

                # activation function
                y_predicted = ActivateSoftmaxSigmoid(z)  # y_predicted = torch.sigmoid(z)  # n*2

                # loss function
                criterion, gradients = CrossEntropy(y_predicted, data_label[batch_ids], data_features[batch_ids])
                if punishEnum is not None:
                    re_loss, re_grad = Regularization(parameters, punishEnum, punish_ratio)
                else:
                    re_loss = 0
                    re_grad = 0

                criterion += re_loss  # n*1
                gradients += re_grad  # n*len(y)*x
                loss = np.sum(criterion) / (end - start)  # 返回的为数值
                batch_loss.append(loss)
                grad = np.sum(gradients, axis=0) / (end - start)
                parameters_new = parameters - learn_ratio * grad.T  # parameters 纵向 x*len(y)
                if n_samples == 2:
                    parameters_new[:, 1] = 0
                # if np.linalg.norm(parameters_new - parameters) / parameters.size < tolerate:
                #     break
                parameters = copy.deepcopy(parameters_new)
    else:
        return

    parameters = np.transpose(parameters)
    # write parameters to metrics to json
    if n_labels == 2:  # binary classification
        # 默认 bias 放在metrics的最后 w0*x0+ ... + w_(n_features-1)*x_(n_features-1) + b*x_(n_features)
        # updated weights and biases in map<map<>>
        i = 0
        for y_i, x_j in metrics_updated.items():
            for j, k in x_j.items():
                metrics_updated[y_i][j] = parameters[0][i]
                i += 1
        # insert position = -1, is second to last
    elif n_labels > 2:  # multi-classification
        t = 0
        for y_i, x_j in metrics.items():
            i = 0
            for j, k in x_j.items():
                metrics_updated[y_i][j] = parameters[t][i]
                i += 1
            t += 1
    else:
        return

    dict_intialtmp = {
        "jobHistoryId": jobhistoryid,
        "globalIter": global_round,
        "count": n_samples,
        "metrics": metrics_updated
    }
    return requests.post(callBackUrl, json=dict_intialtmp)


def LocalTrainLogR(processHistoryId, jobHistoryId, globalRound, metrics, trainFilePath, testFilePath):

    jobhistoryid = jobHistoryId
    global_round = globalRound
    metrics_updated = copy.deepcopy(metrics)

    path_callbackurl = os.path.join('root', "federal", str(processHistoryId), 'callbackurl' + '.json')
    with open(path_callbackurl, "r") as f:
        json_callbackurl = json.load(f)  # _io.TextIOWrapper
    callBackUrl = json_callbackurl['callBackUrl']

    path_config = os.path.join('root', "federal", str(processHistoryId), 'config' + '.json')
    with open(path_config, "r") as f:
        json_conf = json.load(f, indent=4)
    punishEnum = json_conf['punishEnum']
    punish_ratio = json_conf['punish_ratio']  # weight_decay
    optimizeEnum = json_conf['optimizeEnum']
    batchSize = json_conf['batchSize']
    learn_ratio = json_conf['learn_ratio']
    local_ep = json_conf['local_ep']
    classificationEnum = json_conf['classificationEnum']  # ovr

    path_train = os.path.join('root', "federal", str(processHistoryId), 'train_data' + '.csv')
    path_test = os.path.join('root', "federal", str(processHistoryId), 'test_data' + '.csv')
    if not os.path.samefile(path_test, testFilePath):
        return
    elif not os.path.samefile(path_train, trainFilePath):
        return
    data_train = pd.read_csv(path_train, header=0, index_col=0)
    data_test = pd.read_csv(path_test, header=0, index_col=0)
    n_samples = len(data_train)  # the number of the client's data_train(the number of id), used for averaging
    n_labels = len(metrics)  # the number of multi_classes to determine loss function: sigmod or softmax
    n_features = len(data_train.columns)  # the number of features to determine the number of parameters + bias
    # n_features = len(list(metrics['x1'].keys()))  # Python 3x0

    # one-hot encoding & get data
    train_label = OneHotEncoder(data_train['y'].astype(str), metrics)  # data_train['y']为Series可enumerate遍历 return numpy
    train_features = data_train.drop('y', axis=1)
    train_features.insert(len(train_features.columns), '1', np.ones((n_samples, 1)))  # n*x
    train_features = np.array(train_features)
    test_label = OneHotEncoder(data_test['y'].astype(str), metrics)  # data_train['y']为Series可enumerate遍历 return numpy
    test_features = data_test.drop('y', axis=1)
    test_features.insert(len(test_features.columns), '1', np.ones((n_samples, 1)))  # n*x
    test_features = np.array(test_features)

    if not classificationEnum == 'ovr':
        return

    if n_labels == 2:  # binary classification
        parameters = []
        for y_i, x_j in metrics.items():
            for j, k in x_j.items():
                parameters.append(k)
        parameters = np.array(parameters)  # 横向 1*x
        parameters = parameters.reshape(-1, 1)  # 纵向 x*1
        parameters = np.insert(parameters, 1, np.zeros(len(parameters)), axis=1)
    elif n_labels > 2:  # multi-classification
        parameters = [[] for _ in range(n_labels)]
        i = 0
        for y_i, x_j in metrics.items():
            parameters[i] = []
            for j, k in x_j.items():
                parameters[i].append(k)
            i += 1
        parameters = np.array(parameters)  # 横向 len(y)*x
        parameters = parameters.T  # 纵向 x*len(y) or parameters.T reshape(-1, n_labels)
    else:
        return

    # test accuracy of averaged weights of the last round first
    # linear model
    z = np.dot(test_features, parameters)  # n*2 ndarray
    # activation function
    y_predicted = ActivateSoftmaxSigmoid(z)  # y_predicted = torch.sigmoid(z)  # n*2
    # test accuracy
    score = AccuracyScore(y_predicted, test_label)

    # train this global round
    if optimizeEnum == 'sgd':
        for lc_round in range(local_ep):
            batch_loss = []
            batch_num = int(np.ceil(n_samples / batchSize))
            mix_ids = np.random.permutation(n_samples)
            for lc_iter in range(batch_num):
                start = lc_iter * batchSize
                end = (lc_iter + 1) * batchSize
                if end > n_samples:
                    end = n_samples
                batch_ids = mix_ids[start: end]

                # linear model
                z = np.dot(train_features[batch_ids], parameters)  # n*2 ndarray

                # activation function
                y_predicted = ActivateSoftmaxSigmoid(z)  # y_predicted = torch.sigmoid(z)  # n*2

                # loss function
                criterion, gradients = CrossEntropy(y_predicted, train_label[batch_ids], train_features[batch_ids])
                if punishEnum is not None:
                    re_loss, re_grad = Regularization(parameters, punishEnum, punish_ratio)
                else:
                    re_loss = 0
                    re_grad = 0

                criterion += re_loss  # n*1
                gradients += re_grad  # n*len(y)*x
                loss = np.sum(criterion) / (end - start)  # 返回的为数值
                batch_loss.append(loss)
                grad = np.sum(gradients, axis=0) / (end - start)
                parameters_new = parameters - learn_ratio * grad.T  # parameters 纵向 x*len(y)
                if n_samples == 2:
                    parameters_new[:, 1] = 0
                # if np.linalg.norm(parameters_new - parameters) / parameters.size < tolerate:
                #     break
                parameters = copy.deepcopy(parameters_new)
    else:
        return

    parameters = np.transpose(parameters)
    # write parameters to metrics to json
    if n_labels == 2:  # binary classification
        # 默认 bias 放在metrics的最后 w0*x0+ ... + w_(n_features-1)*x_(n_features-1) + b*x_(n_features)
        # updated weights and biases in map<map<>>
        i = 0
        for y_i, x_j in metrics_updated.items():
            for j, k in x_j.items():
                metrics_updated[y_i][j] = parameters[0][i]
                i += 1
        # insert position = -1, is second to last
    elif n_labels > 2:  # multi-classification
        t = 0
        for y_i, x_j in metrics.items():
            i = 0
            for j, k in x_j.items():
                metrics_updated[y_i][j] = parameters[t][i]
                i += 1
            t += 1
    else:
        return

    dict_tmp = {
        "jobHistoryId": jobhistoryid,
        "globalIter": global_round,
        "count": n_samples,
        "score": score,
        "metrics": metrics_updated
    }
    return requests.post(callBackUrl, json=dict_tmp)


def ServerAggregateAverage():
    return


def EndVerify():
    return
