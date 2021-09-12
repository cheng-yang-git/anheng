# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 3:02 下午
# @Author  : Alioth
# @File    : example_data_split.py
# @Email   : thxthx1999@gmail.com
# @Software: PyCharm


import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import requests

if __name__ == '__main__':

    path_project = os.path.abspath('..')
    path_callbackurl = os.path.join(path_project, 'data', 'callbackurl' + '.json')
    print(path_callbackurl)
    with open(path_callbackurl, "r") as f:
        json_callbackurl = json.load(f)
    callBackUrl = json_callbackurl['callBackUrl']
    print(callBackUrl)

    path_data = os.path.join(path_project, 'data', 'loan_data' + '.csv')
    data_set = pd.read_csv(path_data, header=0, index_col=0)
    # Initiate a list for categoricals
    categ_list = ['purpose']
    # create new df with dummy variables
    data_set = pd.get_dummies(data_set, columns=categ_list, drop_first=True)
    # # print(data_set)
    percent = 0.8
    random = 1234
    test_data = data_set.sample(frac=(1 - percent), replace=False, random_state=random, axis=0)
    train_data = data_set[~data_set.index.isin(test_data.index)]
    print(test_data.head())

    path_train = os.path.join(path_project, 'data', 'train_data' + '.csv')
    path_test = os.path.join(path_project, 'data', 'test_data' + '.csv')

    train_data.to_csv(path_train)
    test_data.to_csv(path_test)

    dict_path = {
        "path_train": path_train,
        "path_test": path_test
    }

    r = requests.post(callBackUrl, json=dict_path)  # does json.dumps(your_json) automatically
