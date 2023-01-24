import numpy as np
import matplotlib.pyplot as plt
from srcs.network import Network
from srcs.fc_layer import FCLayer
from srcs.activation_layer import ActivationLayer
from srcs.activations import tanh, tanh_prime, sigmoid, sigmoid_prime
from srcs.loss_functions import mse, mse_prime


from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Before traitement: ")
print(f"shape of x_train = {x_train.shape} et shape of y_train = {y_train.shape}")

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
print("After traitement: ")

# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

print(f"shape of x_train = {x_train.shape} et shape of y_train = {y_train.shape}")

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
# print(x_test[0].shape)
# print(x_test[0])
y_test = np_utils.to_categorical(y_test)

# Network   
net = Network()
net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(mse, mse_prime)
print("training...", end='')
err = net.fit(x_train[0:1000], y_train[0:1000], epochs=40, learning_rate=0.1)
print(" OK")

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.arange(len(err)), err)
ax.set_xlabel("epoch")
ax.set_ylabel("mse")
plt.show()


out = net.predict(x_test)
good = 0
for o,t in zip(out, y_test):
    if np.argmax(o) == np.argmax(t):
        good += 1
print(f"good = {good} % = {good / len(y_test) * 100}")

ending = False
while not ending:
    num = input(f"Enter a number [0,{len(y_test) - 1}]: ")
    try:
        num = int(num)
    except ValueError:
        break
    if num > len(y_test):
        break
    out = net.predict(x_test[num])
    prediction = np.argmax(out)
    trueValue = np.argmax(y_test[num])
    print(f"P = {prediction}\tT = {trueValue} -> ",end='')
    if prediction == trueValue:
        print("OK")
    else:
        print("KO")
    plt.imshow(x_test[num].reshape(28, 28))
    plt.show()
print("Good By !")