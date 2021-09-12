# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/2 4:27 下午
# @Author  : Alioth
# @File    : feature_scaling.py
# @Email   : thxthx1999@gmail.com
# @Software: PyCharm


import os
import pandas as pd
from sklearn import preprocessing


def FeatureScaling(processHistoryId, jobHistoryId, functionEnum):

    jobhistoryid = jobHistoryId

    # 对训练集标准化/归一化
    path_train = os.path.join('root', "federal", str(processHistoryId), 'train_data' + '.csv')
    path_test = os.path.join('root', "federal", str(processHistoryId), 'test_data' + '.csv')
    data_train = pd.read_csv(path_train, header=0, index_col=0)
    data_test = pd.read_csv(path_test, header=0, index_col=0)
    id_train = data_train.index.values
    id_test = data_test.index.values
    # if set(['A', 'C']).issubset(df.columns) or {'A', 'C'}.issubset(df.columns):
    if 'y' in data_train.columns:
        label_train = data_train['y']
        label_test = data_test['y']
        features_train = data_train.drop('y', axis=1)  # data_train do not change
        features_test = data_test.drop('y', axis=1)
        columns_train = features_train.columns.values
        columns_test = features_test.columns.values
        if functionEnum == "z-score":
            scaler = preprocessing.StandardScaler()
            features_train_scaled = scaler.fit_transform(features_train)
            features_test_scaled = scaler.transform(features_test)
        elif functionEnum == "min-max":
            scaler = preprocessing.MinMaxScaler()
            features_train_scaled = scaler.fit_transform(features_train)
            features_test_scaled = scaler.transform(features_test)
        else:
            features_train_scaled = features_train
            features_test_scaled = features_test
        features_train_scaled = pd.DataFrame(features_train_scaled)
        features_test_scaled = pd.DataFrame(features_test_scaled)
        features_train_scaled.columns = columns_train
        features_test_scaled.columns = columns_test
        features_train_scaled.insert(0, 'id', id_train)
        features_train_scaled = features_train_scaled.set_index('id')
        features_test_scaled.insert(0, 'id', id_test)
        features_test_scaled = features_test_scaled.set_index('id')
        features_train_scaled.insert(len(features_train_scaled), 'y', label_train)
        features_test_scaled.insert(len(features_test_scaled), 'y', label_test)
    else:
        features_train = data_train
        columns_train = features_train.columns.values
        features_test = data_test.drop('y', axis=1)
        columns_test = features_test.columns.values
        if functionEnum == "z-score":
            scaler = preprocessing.StandardScaler()
            features_train_scaled = scaler.fit_transform(features_train)
            features_test_scaled = scaler.transform(features_test)
        elif functionEnum == "min-max":
            scaler = preprocessing.MinMaxScaler()
            features_train_scaled = scaler.fit_transform(features_train)
            features_test_scaled = scaler.transform(features_test)
        else:
            features_train_scaled = features_train
            features_test_scaled = features_test
        features_train_scaled = pd.DataFrame(features_train_scaled)
        features_train_scaled.columns = columns_train
        features_test_scaled.columns = columns_test
        features_train_scaled.insert(0, 'id', id_train)
        features_train_scaled = features_train_scaled.set_index('id')
        features_test_scaled.insert(0, 'id', id_test)
        features_test_scaled = features_test_scaled.set_index('id')

    features_train_scaled.to_csv(path_train)
    features_test_scaled.to_csv(path_test)

    return jobhistoryid
