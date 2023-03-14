import getopt
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from srcs.utils_ml import prepare_data, category_to_bool
from srcs.metrics import perf_measure, f1_score_
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
    print("\t\t(without option, the prediction is done with the best model )")
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
        print("-------------------------------------------------------------------------------")
        print("The 'neural_network_params.yml' is formated like this : ")
        print("networks:")
        print("  - name: model.pkl")
        print("    layers:")
        print("      - neurons: 30")
        print("        activation: relu")
        print("      - neurons: 15")
        print("        activation: relu")
        print("      - neurons: 2")
        print("        activation: softmax")
        print("    epoch: 2000")   
        print("\nwith Activations : relu, softmax, sigmoid, tanh")
    exit(1)

def print_succes(out, y_test, model):
    """
        print the performances of the model on the test data
    """
    print(f"\t using the model : {model}")
    good = 0
    for o,t in zip(out, y_test):
        if np.argmax(o) == t:
            good += 1
    print(f"Succes = {good / len(y_test) * 100:.2f}%\t err = {100 - (good / len(y_test) * 100):.2f}%")
    f1_score = f1_score_(y_test, category_to_bool(out))
    print(f"f1 score = {colors.green}{f1_score:.4f}{colors.reset}")
    confusion = confusion_matrix_(y_true= y_test, y_hat=category_to_bool(out), df_option=True)
    print(confusion)
    TP, FP, TN, FN = perf_measure(y=y_test, y_hat=category_to_bool(out))
    print(f"{TN + FN} {colors.green}begnin{colors.reset} cells with {TN} Thrue and {FN} False")
    print(f"{TP+FP} {colors.red}malignant{colors.reset} cells with {TP} True and {FP} False")
    print(f"False Positive = {colors.red}{FP}{colors.reset}\tFalse Negative = {colors.red}{FN}{colors.reset}")

def loop_multi_training(data, split, verbose=False, graphics=False):
    """
    Training all the model in models/neural_network_params.yml
    """
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
            lns1 = ax1.plot(np.arange(len(loss)), loss, label='Cross-Entropy', color='r')
            lns2 = ax2.plot(np.arange(len(accuracy)), accuracy, label='Accuracy', color='b')
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc=0)
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("Cross-Entropy")
            ax2.set_ylabel("Accuracy")
            plt.title(file_name)
        print("-----------------------------------------------------------")
    plt.show()

def loop_train(data, split, verbose, model_name, graphics=False):
    """
        train the model_name only
    """
    x_train, y_train, x_test, y_test = prepare_data(data, verbose, split)
    model = get_model('models/neural_network_params.yml', model_name=model_name)
    print(f"\t model : {model}")
    model._compile(x_train)

    loss, accuracy = model.train(x_train, y_train, model.epochs, verbose)

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
        lns1 = ax1.plot(np.arange(len(loss)), loss, label='Cross-Entropy', color='r')
        lns2 = ax2.plot(np.arange(len(accuracy)), accuracy, label='Accuracy', color='b')
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("Cross-Entropy")
        ax2.set_ylabel("Accuracy")
        plt.title(file_name)
        plt.show()
    if verbose:
        print(f"Prediction on datatest ({len(x_test)} lines)")
        out = model.predict(x_test)
        print_succes(out, y_test, model)

def predict(data, file_model, verbose, split):
    """
    make prediction with the file_model
    """
    _, _, x_test, y_test = prepare_data(data, verbose, split)
    model = None
    with open(file_model, "rb") as f:
        model = pickle.load(f)
    print(f"Prediction on datatest ({len(x_test)} lines)")  
    out = model.predict(x_test)
    print_succes(out, y_test, model)

def best(verbose=False):
    """
    return the best Model
    """
    tab_models = load_models('models/neural_network_params.yml')
    best_accuracy = 0
    best_model_accuracy = None
    for model in tab_models:
        if verbose:
            print(model)
        cross = model.get_cross_entropy()
        accuracy = model.get_accuracy()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_accuracy = model
    if best_model_accuracy is not None:
        if verbose:
            print(f"Best Model is {colors.green}{best_model_accuracy.file}{colors.reset} with Accuracy = {colors.blue}{best_model_accuracy.get_accuracy()}{colors.reset}")
        return f"models/{best_model_accuracy.file}"
    else:
        if verbose:
            print(f"Models {colors.red}Not Found{colors.reset}")
        return None

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
                best(True)
                return
        if mode not in ["train", "predict",]:
            usage("Bad Mode")
        print(f"********** {colors.green}{mode.upper()}{colors.reset} **********")
        if mode == "train":
            if model is not None:
                loop_train(data=data, split=split, verbose=verbose, model_name=model, graphics=graphics)
            else:
                loop_multi_training(data=data, split=split, verbose=verbose, graphics=graphics)
        else: # Predict mode
            if model is None:
                model = best(verbose=verbose)
                if model is None:
                    usage("no model has been trained")
            predict(data=data,file_model=model, verbose=verbose, split=split)
        
    except Exception as inst:
        usage(inst)

if __name__ == "__main__":
    main(sys.argv[1:])
    print("Good by !")