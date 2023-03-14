import getopt
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from srcs.utils_ml import data_spliter, normalize
# from srcs.network import Network
from srcs.metrics import f1_score_, perf_measure
# from srcs.layer import DenseLayer
from srcs.confusion_matrix import confusion_matrix_
from srcs.common import colors, load_data
from srcs.yml_utils import create_models, get_model, load_models

import pickle

def usage(string = None, extend=False):
    if string is not None:
        print(f"{colors.red}{string}{colors.reset}")
    print("usage: main --file=DATA (--train|--predict) [--model=] [--split=] [--verbose] [--graphics] [--help]")
    print("\t-f | --file=  : 'dataset.csv'")
    print("\t-t | --train  : training mode ( the option --model= can be a specific model describe in models/neural_netk_params.yml)")
    print("\t-p | --predict: predict mode (the --model option can be the file of the model to predict)")
    print("\t-o | --model= model's name : name of the model to load in training mode (default models/model_xx.pkl)")
    print("\t\t or a neural networks params.yml file in training mode")
    print("\t-s | --split= xx.yy : split ratio between training and test data: (--split=0.7 means 70% for training and 30% for test) (default 0.8)")
    print("\t-v | --verbose : mode Verbose to print for each epochs the loss and accuracy")
    print("\t-g | --graphics : drawing graphics in training modes (default: False)")
    print("\t-h | --help : help page.")
    if extend:
        print("\n\Examples of uses :")
        print("\t$> python3 main.py -f dataset/data.csv --train --model=model_2.pkl")
        print("\t$> python3 main.py -f dataset/data.csv -t -o model_2.pkl -s 0.7")
        print("\t$> python3 main.py -f dataset/data.csv --predict -o models/model.pkl")
        print("\t$> python3 main.py -f dataset/data.csv -p -o models/model_2.pkl")
    exit(1)

def fit_transform(targets:np.ndarray, labels:list):
    """
    the fit_transform of LabelEncord from sklearn.preprocessing
    in binaray option given by labels
    """
    if len(labels) != 2:
        raise("Error in fit_transform: bad len of labels")
        return None
    res = np.zeros((targets.shape[0]), dtype=int)
    for idx, target in enumerate(targets):
        res[idx] = (target[0] == labels[0])
    return res

def print_succes(out, y_test, model):
    print(f"\t using the model : {model}")
    good = 0
    for o,t in zip(out, y_test):
        if np.argmax(o) == t:
            good += 1
    print(f"Succes = {good / len(y_test) * 100:.2f}%\t err = {100 - (good / len(y_test) * 100):.2f}%")
    confusion = confusion_matrix_(y_true= y_test, y_hat=category_to_bool(out), df_option=True)
    print(confusion)
    TP, FP, TN, FN = perf_measure(y=y_test, y_hat=category_to_bool(out))
    print(f"{TN + FN} {colors.green}begnin{colors.reset} cells with {TN} Thrue and {FN} False")
    print(f"{TP+FP} {colors.red}malignant{colors.reset} cells with {TP} True and {FP} False")
    print(f"False Positive = {colors.red}{FP}{colors.reset}\tFalse Negative = {colors.red}{FN}{colors.reset}")


def category_to_bool(arr:np.ndarray):
    res = np.zeros(len(arr))
    for idx, el in enumerate(arr):
        res[idx] = np.argmax(el)
    return res

def prepare_data(data, verbose, split=0.8):
    """
        Prepare the data :
            - split in x_train, y_train, x_test, y_test with split param
            - Normalize
            - put 'M' & 'B' on 1|0
    """
    target = np.array(data[1].values).reshape(-1,1)
    Xs = np.array(data[data.columns[2:]].values)
    Xs=normalize(Xs)
    #split data
    x_train, y_train, x_test, y_test = data_spliter(Xs, target, split)

    y_train = fit_transform(y_train, ['M', 'B'])
    y_test = fit_transform(y_test, ['M', 'B'])
    
    if verbose:
        nb_input = x_train.shape[1]
        nb_train = x_train.shape[0]
        nb_test = x_test.shape[0]
        print("*** Preparation of Data ***")
        print(f"\tNb of data : {colors.green}{len(Xs)}{colors.reset}")
        print(f"\tNb features : {colors.blue}{nb_input}{colors.reset}")
        print(f"\tNb data for training ({colors.yellow}{split*100}%{colors.reset}) : {colors.blue}{nb_train}{colors.reset}")
        print(f"\tNb data for test ({colors.yellow}{100-(split*100)}%{colors.reset}) : {colors.blue}{nb_test}{colors.reset}")
        print("*******************")
    return x_train, y_train, x_test, y_test

def loop_multi_training(data, split, verbose=False, graphics=False):
    x_train, y_train, _, _ = prepare_data(data, verbose, split)
    tab_models = create_models('models/neural_network_params.yml')
    for model in tab_models:
        print(f"\t model : {model}")
        model._compile(x_train)
        loss, accuracy = model.train(x_train, y_train, model.epochs, verbose)
        file_name = f"models/{model.file}"
        print(f"save model in {colors.blue}{file_name}{colors.reset} ...", end="")
        with open(file_name, "wb") as f:
            pickle.dump(model, f)
        print(f"{colors.green}OK{colors.reset}")
        print(f"Epochs = {colors.blue}{model.epochs}{colors.reset}, Cross-Entropy = {colors.blue}{loss}{colors.reset}, Accuracy = {colors.blue}{accuracy}{colors.reset}")
        accuracy = model.accuracy
        loss = model.loss

        if graphics:
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
            plt.title(file_name)
    plt.show()

def loop_train(data, split, verbose, model_name, graphics=False):

    x_train, y_train, x_test, y_test = prepare_data(data, verbose, split)
    model = get_model('models/neural_network_params.yml', model_name=model_name)
    print(f"\t model : {model}")
    model._compile(x_train)

    loss, accuracy = model.train(x_train, y_train, 6000, verbose)

    # sauvegarde 
    file_name = f"models/{model.file}"
    
    print(f"save model in {colors.blue}{file_name}{colors.reset} ...", end="")
    with open(file_name, "wb") as f:
        pickle.dump(model, f)
        print(f"{colors.green}OK{colors.reset}")
    print(f"Cross-Entropy = {colors.blue}{loss}{colors.reset}, Accuracy = {colors.blue}{accuracy}{colors.reset}")
    accuracy = model.accuracy
    loss = model.loss
    if graphics:
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
        plt.title(file_name)
        plt.show()
    if verbose:
        print(f"Prediction on datatest ({len(x_test)} lines)")
        out = model.predict(x_test)
        print_succes(out, y_test, model)

def predict(data, file_model, verbose, split):
    _, _, x_test, y_test = prepare_data(data, verbose, split)
    model = None
    with open(file_model, "rb") as f:
        model = pickle.load(f)
    print(f"Prediction on datatest ({len(x_test)} lines)")  
    out = model.predict(x_test)
    print_succes(out, y_test, model)

def best():
    """
    return name of best Model
    """
    tab_models = load_models('models/neural_network_params.yml')
    best_cross = 100
    best_model = None
    for model in tab_models:
        print(model)
        cross = model.get_cross_entropy()
        if cross < best_cross:
            best_cross = cross
            best_model = model
    if best_model is not None:
        print(f"Best Model is {colors.green}{best_model.file}{colors.reset} with Cross-Entropy = {colors.blue}{best_model.get_cross_entropy()}{colors.reset}")
    else:
        print(f"Best Model {colors.red}Not Found{colors.reset}")
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:m:o:s:vhtpgb", ["file=", "predict", "train", "model=", "help", "split=", "verbose", "graphics", "best"])
    except getopt.GetoptError as inst:
        usage(inst)

    try:
        mode = None
        data = None
        verbose = False
        model = None
        graphics = False
        split = 0.8
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                data = load_data(arg, header=None)
        if data is None:
            usage("No data")
        for opt, arg in opts:
            if opt in ["-t", "--train"]:
                mode = "train"
            elif opt in ["-p", "--predict"]:
                mode = "predict"
            elif opt in ["-v", "--verbose"]:
                verbose = True
            elif opt in ["-o", "--model"]:
                model = arg
            elif opt in ["-h", "--help"]:
                usage(extend=True)
            elif opt in ["-g", "--graphics"]:
                graphics = True
            elif opt in ["-s", "--split"]:
                split = float(arg)
            elif opt in ["-b", "--best"]:
                best()
                return
        if mode not in ["train", "predict"]:
            usage("Bad Mode")
        print(f"********** {colors.green}{mode.upper()}{colors.reset} **********")
        if mode == "train":
            if model is not None:
                loop_train(data=data, split=split, verbose=verbose, model_name=model, graphics=graphics)
            else:
                loop_multi_training(data=data, split=split, verbose=verbose, graphics=graphics)
        else: # Predict mode
            if model is None:
                model = "models/model.pkl"
            predict(data=data,file_model=model, verbose=verbose, split=split)
        
    except Exception as inst:
        usage(inst)

if __name__ == "__main__":
    main(sys.argv[1:])
    print("Good by !")