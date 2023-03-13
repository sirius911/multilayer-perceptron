import getopt
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from srcs.utils_ml import data_spliter, normalize
from srcs.network import Network
from srcs.metrics import f1_score_, perf_measure
from srcs.layer import DenseLayer
from srcs.confusion_matrix import confusion_matrix_
from srcs.common import colors, load_data, error
from srcs.header import header
import pickle

def usage(string = None):
    if string is not None:
        print(string)
    print("usage: main --file=DATA --mode=[train | predict] [--verbose]")
    print("\t-f | --file=  : 'dataset.csv'")
    print("\t-t | --mode=  : mode of function (train | predict)")
    print("\t-v | --verbose : mode Verbose to print for each epochs the loss and accuracy")
    exit(1)

def fit_transform(targets:np.ndarray, labels:list):
    """
    the fit_transform of LabelEncord from sklearn.preprocessing
    in binaray option given by labels
    """
    if len(labels) != 2:
        raise("Error in fit_transform: bad lenof labels")
        return None
    res = np.zeros((targets.shape[0]), dtype=int)
    for idx, target in enumerate(targets):
        res[idx] = (target[0] == labels[0])
    return res

def category_to_bool(arr:np.ndarray):
    res = np.zeros(len(arr))
    for idx, el in enumerate(arr):
        res[idx] = np.argmax(el)
    return res

def loop_train(data, verbose):
    # data = pd.read_csv("dataset/data.csv", header=None)
    target = np.array(data[1].values).reshape(-1,1)

    # M -> 1
    # B -> 0

    # y = np.array(LabelEncoder().fit_transform(target).reshape(-1, 1))
    # print(f"y[:3]={y[:3]}")
    # print(f"y.shape = {y.shape}")

    Xs = np.array(data[data.columns[2:]].values)
    Xs=normalize(Xs)

    #split data
    x_train, y_train, x_test, y_test = data_spliter(Xs, target, 0.8)

    y_train = fit_transform(y_train, ['M', 'B'])
    y_test = fit_transform(y_test, ['M', 'B'])

    # print(type(Xs), Xs.shape, type(x_train), x_train.shape)
    # print(type(Xs[0][0]), type(x_train[0][0]))

    nb_input = x_train.shape[1]
    print(f"Shape of Xs = {Xs.shape} and shape of y = {y_train.shape}")
    print(f"Nb of data : {colors.green}{len(Xs)}{colors.reset}")
    print(f"\tNb labels : {colors.blue}{nb_input}{colors.reset}")

    model = Network()
    model.add(DenseLayer(75, 'relu'))
    model.add(DenseLayer(35, 'relu'))
    model.add(DenseLayer(15, 'relu'))
    model.add(DenseLayer(2, 'softmax'))

    model._compile(x_train)

    loss, accuracy = model.train(x_train, y_train, 6000, verbose)

    # sauvegarde
    print("save model in models/model.pkl ...", end="")
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
        print(f"{colors.green}OK{colors.reset}")
    print(f"Cross-Entropy = {colors.blue}{loss}{colors.reset}, Accuracy = {colors.blue}{accuracy}{colors.reset}")
    accuracy = model.accuracy
    loss = model.loss

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax2 = ax1.twinx()
    lns1 = ax1.plot(np.arange(len(loss)), loss, label='loss', color='r')
    lns2 = ax2.plot(np.arange(len(accuracy)), accuracy, label='Accuracy', color='b')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("Accuracy")
    plt.show()

def predict(data):
    target = np.array(data[1].values).reshape(-1,1)
    Xs = np.array(data[data.columns[2:]].values)
    Xs=normalize(Xs)
    _, _, x_test, y_test = data_spliter(Xs, target, 0.8)

    y_test = fit_transform(y_test, ['M', 'B'])
    model = None
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    print(f"Prediction on datatest ({len(x_test)} lines)")
    print(model.loss[-1])    
    out = model.predict(x_test)

    good = 0
    for o,t in zip(out, y_test):
        if np.argmax(o) == t:
            good += 1
    print(f"Succes = {good / len(y_test) * 100:.2f}%\t err = {100 - (good / len(y_test) * 100):.2f}%")
    f1 = f1_score_(y=y_test, y_hat= category_to_bool(out))
    print(f"F1 score = {colors.green}{f1}{colors.reset}")
    cross_entropy = model._calculate_loss(out, y_test)
    print(f"Cross-Entropy = {cross_entropy}")
    confusion = confusion_matrix_(y_true= y_test, y_hat=category_to_bool(out), df_option=True)
    print(confusion)
    TP, FP, TN, FN = perf_measure(y=y_test, y_hat=category_to_bool(out))
    print(f"{TN + FN} {colors.green}begnin{colors.reset} cells with {TN} Thrue and {FN} False")
    print(f"{TP+FP} {colors.red}malignant{colors.reset} cells with {TP} True and {FP} False")
    print(f"False Positive = {colors.red}{FP}{colors.reset}\tFalse Negative = {colors.red}{FN}{colors.reset}")

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:m:v", ["file=", "mode=", "verbose"])
    except getopt.GetoptError as inst:
        usage(inst)

    try:
        mode = None
        data = None
        verbose = False
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                data = load_data(arg, header=None)
        if data is None:
            usage("No data")
        for opt, arg in opts:
            if opt in ["-m", "--mode"]:
                mode = arg
            elif opt in ["-v", "--verbose"]:
                verbose = True
        if mode not in ["train", "predict"]:
            usage("Bad Mode")
        print(f"********** {colors.green}{mode.upper()}{colors.reset} **********")
        if mode == "train":
            loop_train(data, verbose)
        else:
            predict(data)
        print("Good by !")
    except Exception as inst:
        usage(inst)

if __name__ == "__main__":
    main(sys.argv[1:])