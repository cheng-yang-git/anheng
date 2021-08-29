import os
import json
import numpy as np

def get_data(id=""):
    path_train = os.path.join("FLdata", "train", "mnist_train_" + str(id) + ".json")
    path_test = os.path.join("FLdata", "test", "mnist_test_" + str(id) + ".json")
    data_train = {}
    data_test = {}

    with open(os.path.join(path_train), "r") as f_train:
        train = json.load(f_train)
        data_train.update(train['user_data'])
    with open(os.path.join(path_test), "r") as f_test:
        test = json.load(f_test)
        data_test.update(test['user_data'])

    X_T, y_T, X_t, y_t = data_train['0']['x'], data_train['0']['y'], data_test['0']['x'], data_test['0']['y']
    y_T = [int(x) for x in y_T]
    y_t = [int(x) for x in y_t]
    num_T, num_t = len(y_T), len(y_t)
    return np.array(X_T), np.array(y_T), np.array(X_t), np.array(y_t), num_T, num_t

def softmax(X, W):  
    x = X.dot(W)
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax

def loss(X, y, W):
    P = softmax(X, W)
    n = range(X.shape[0])
    return -np.mean(np.log(P[n, y]))  

def grad(X, y, W):
    pred = softmax(X, W)
    x_range = range(X.shape[0])  
    pred[x_range, y] -= 1 
    return X.T.dot(pred) / X.shape[0]   

def simple_lr(X, y, W, eta, E, batch_size, t, tol=1e-5):
    W_old = W.copy()
    ep = 0
    loss_hist = []  
    N = X.shape[0]
    nbatches = int(np.ceil(float(N) / batch_size))

    while ep < E:
        ep += 1
        mix_ids = np.random.permutation(N)  
        for i in range(nbatches):
            batch_ids = mix_ids[batch_size * i:min(batch_size * (i + 1), N)]
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W -= eta / np.sqrt(t) * grad(X_batch, y_batch, W)
        loss_hist.append(loss(X, y, W))
        if np.linalg.norm(W - W_old) / W.size < tol:  
            break
        W_old = W.copy()
    loss_sum = 0
    for l in loss_hist:
        loss_sum += l
    average_loss = loss_sum/len(loss_hist)
    return W, average_loss

def pred(W, X):
    P = softmax(X, W)
    return np.argmax(P, axis=1)  

def accuracy(y_pre, y):
    count = y_pre == y  
    return count.sum() / len(count) *100