# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 3:03 下午
# @Author  : Alioth
# @File    : example_feature_scaling.py
# @Email   : thxthx1999@gmail.com
# @Software: PyCharm


import os
import pandas as pd
from sklearn import preprocessing

if __name__ == '__main__':

    path_project = os.path.abspath('..')
    path_train = os.path.join(path_project, 'data', 'train_data' + '.csv')
    path_test = os.path.join(path_project, 'data', 'test_data' + '.csv')
    data_train = pd.read_csv(path_train, header=0, index_col=0)
    data_test = pd.read_csv(path_test, header=0, index_col=0)
    functionEnum = 'z-score'
    id_train = data_train.index.values
    id_test = data_test.index.values
    if 'not.fully.paid' in data_train.columns:
        label_train = data_train['not.fully.paid']
        features_train = data_train.drop('not.fully.paid', axis=1)  # data_train do not change
        columns_train = features_train.columns.values
        label_test = data_test['not.fully.paid']
        features_test = data_test.drop('not.fully.paid', axis=1)  # data_train do not change
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
        features_train_scaled.insert(0, 'id', id_train)
        features_train_scaled = features_train_scaled.set_index('id')
        features_train_scaled.insert(len(features_train_scaled.columns), 'not.fully.paid', label_train)
        features_test_scaled = pd.DataFrame(features_test_scaled)
        features_test_scaled.columns = columns_test
        features_test_scaled.insert(0, 'id', id_test)
        features_test_scaled = features_test_scaled.set_index('id')
        features_test_scaled.insert(len(features_test_scaled.columns), 'not.fully.paid', label_test)
    else:
        features_train = data_train
        columns_train = features_train.columns.values
        if functionEnum == "z-score":
            scaler = preprocessing.StandardScaler()
            features_train_scaled = scaler.fit_transform(features_train)
        elif functionEnum == "min-max":
            scaler = preprocessing.MinMaxScaler()
            features_train_scaled = scaler.fit_transform(features_train)
        else:
            features_train_scaled = features_train
        features_train_scaled = pd.DataFrame(features_train_scaled)
        features_train_scaled.columns = columns_train
        features_train_scaled.insert(0, 'id', id_train)
        features_train_scaled = features_train_scaled.set_index('id')

    print(features_train_scaled.head())
    print(features_test_scaled.head())

    path_train_scaled = os.path.join(path_project, 'data', 'train_data_scaled' + '.csv')
    features_train_scaled.to_csv(path_train_scaled)
    path_test_scaled = os.path.join(path_project, 'data', 'test_data_scaled' + '.csv')
    features_test_scaled.to_csv(path_test_scaled)
