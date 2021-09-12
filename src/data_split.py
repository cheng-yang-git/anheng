# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/2 4:26 下午
# @Author  : Alioth
# @File    : data_split.py
# @Email   : thxthx1999@gmail.com
# @Software: PyCharm

import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import requests


def DataSplit(processHistoryId, jobHistoryId, path, random, percent=0.8):

    # 获取回调url
    path_callbackurl = os.path.join('root', "federal", str(processHistoryId), 'callbackurl' + '.json')
    with open(path_callbackurl, "r") as f:
        json_callbackurl = json.load(f)  # _io.TextIOWrapper
    callBackUrl = json_callbackurl['callBackUrl']

    # 规定分割数据集后地址
    path_train = os.path.join('root', "federal", str(processHistoryId), 'train_data' + '.csv')
    path_test = os.path.join('root', "federal", str(processHistoryId), 'test_data' + '.csv')

    jobhistoryid = jobHistoryId

    # 默认表头写入文件runtime_params .csv
    data_set = pd.read_csv(path, header=0, index_col=0)
    test_data = data_set.sample(frac=(1 - percent), replace=False, random_state=random, axis=0)
    train_data = data_set[~data_set.index.isin(test_data.index)]

    # 存储分割数据
    train_data.to_csv(path_train)
    test_data.to_csv(path_test)

    # 回调http接口
    dict_path = {
        "path_train": "'" + path_train + "'",
        "path_test": "'" + path_test + "'"
    }
    r = requests.post(callBackUrl, json=dict_path)  # does json.dumps(your_json) automatically

    return jobhistoryid, path_train, path_test
