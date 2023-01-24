import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from srcs.utils_ml import data_spliter
from srcs.network import Network
from srcs.fc_layer import FCLayer
from srcs.activation_layer import ActivationLayer
from srcs.activations import tanh, tanh_prime, sigmoid, sigmoid_prime
from srcs.loss_functions import mse, mse_prime
from srcs.metrics import f1_score_

def to_categorical(arr:np.ndarray, labels:list):
    res = np.zeros((arr.shape[0], len(labels)))
    for e, li in zip(arr, res):
        for idx,label in enumerate(labels):
            # print(f"e = {e} label = {label}")
            if e == label:
                li[idx] = 1
    #     print(li)
    # print(res)
    return res

def category_to_bool(arr:np.ndarray):
    res = np.zeros(len(arr))
    for idx, el in enumerate(arr):
        res[idx] = np.argmax(el)
    return res

data = pd.read_csv("data.csv", header=None)
target = np.array(data[1].values).reshape(-1, 1)
Xs = np.array(data[data.columns[2:]].values)
nb_input = Xs.shape[1]
print(f"Nb of data : {len(Xs)}")
print(f"\tNb input : {nb_input}")
print(f"Shape of Xs = {Xs.shape} and shape of target = {target.shape}")
#split data
x_train, y_train, x_test, y_test = data_spliter(Xs, target, 0.8)
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_train = x_train.astype('float32')
y_train = to_categorical(y_train, ['M', 'B'])
print(f"shape of x_train = {x_train.shape} et shape of y_train = {y_train.shape}")

#same with test data
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
x_test = x_test.astype('float32')
y_test = to_categorical(y_test, ['M', 'B'])

#network
net = Network()
net.add(FCLayer(nb_input, 100))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 2))
net.add(ActivationLayer(sigmoid, sigmoid_prime))

net.use(mse, mse_prime)
print("training ...")
err, f1 = net.fit(x_train, y_train, epochs=40, learning_rate=0.1)
print("ok")
fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()
lns1 = ax1.plot(np.arange(len(err)), err, label='loss', color='r')
lns2 = ax2.plot(np.arange(len(f1)), f1, label='f1', color='b')
lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax2.set_ylabel("f1 score")
plt.show()

out = net.predict(x_test)
good = 0
for o,t in zip(out, y_test):
    if np.argmax(o) == np.argmax(t):
        good += 1
print(f"good = {good} ==> {good / len(y_test) * 100:.2f}%")
y_hat = category_to_bool(np.array(out).reshape(len(out),2))
print(f"f1_score = {f1_score_(category_to_bool(y_test), y_hat):0.2f}")