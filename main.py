import getopt
import copy
import sys
import numpy as np
from srcs.utils_ml import prepare_data, category_to_bool, prepare_cross_data, cross_validation
from srcs.metrics import perf_measure, f1_score_
from srcs.confusion_matrix import confusion_matrix_
from srcs.common import colors, load_data
from srcs.yml_utils import create_models, get_model, load_models
from srcs.graph import graph, draw_matrix_confusion
from tqdm import tqdm

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
    print("\t-k | --K=xx : Split the dataset in xx parts for K-folds Validation (Default = 10)")
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

def print_succes(out, y_test, model, graphics=None):
    """
        print the performances of the model on the test data
    """
    print(f"\t using the model : {model}")
    good = 0
    for o,t in zip(out, y_test):
        if np.argmax(o) == t:
            good += 1
    error = 100 - (good / len(y_test) * 100)
    print(f"Succes = {colors.green}{good / len(y_test) * 100:.2f}%{colors.reset}\t err = {colors.red}{error:.2f}%{colors.reset}")
    f1_score = f1_score_(y_test, category_to_bool(out))
    print(f"f1 score = {colors.green}{f1_score:.4f}{colors.reset}")
    TP, FP, TN, FN = perf_measure(y=y_test, y_hat=category_to_bool(out))
    print(f"{TN + FN} {colors.green}begnin{colors.reset} cells with {TN} Thrue and {FN} False")
    print(f"{TP+FP} {colors.red}malignant{colors.reset} cells with {TP} True and {FP} False")
    print(f"False Positive = {colors.red}{FP}{colors.reset}\tFalse Negative = {colors.red}{FN}{colors.reset}")
    title = f"{model.file} - f1 = {f1_score:.4f} - error = {error:.2f}%"
    if graphics is not None:
        draw_matrix_confusion(confusion_matrix=confusion_matrix_(y_true= y_test, y_hat=category_to_bool(out), df_option=False), title=title)

def save_model(model, verbose=False):
    # sauvegarde 
    file_name = f"models/{model.file}"
    if verbose:
        print(f"save model in {colors.blue}{file_name}{colors.reset} ...", end="")
    with open(file_name, "wb") as f:
        pickle.dump(model, f)
        if verbose:
            print(f"{colors.green}OK{colors.reset}")
        
def train(model, x_train, y_train, x_test, y_test, verbose, save):
    model._compile(input = x_train.shape[1])
    model.train(x_train, y_train, model.epochs, verbose)
    #calcul f1_score
    out = model.predict(x_test)
    f1_score = f1_score_(y_test, category_to_bool(out))
    model.f1_score = f1_score
    if verbose:
        print(f"Prediction on datatest ({len(x_test)} lines)")
        print_succes(out, y_test, model)
    return f1_score


def loop_multi_training(data, K, verbose=False, graphics=None):
    """
    Training all the model in models/neural_network_params.yml
    """
    tab_models = create_models('models/neural_network_params.yml')
    for model in tab_models:
        loop_train_cross_data(data=data, K=K, model_empty=model, verbose=verbose, graphics=graphics)
        print("-----------------------------------------------------------")

def loop_train_cross_data(data, K, model_empty, verbose=False, graphics=None):
    """
        train the model_name only
    """

    X, Y = prepare_cross_data(data, K)
    model=None
    f1_score_sum = 0
    best_model = None
    best_score = 0
    for _,k_folds in zip(tqdm(range(K), desc=f"{model_empty.file} - K_folds", colour='blue'), cross_validation(X, Y, K)):
        model = copy.deepcopy(model_empty)
        x_train, y_train, x_test, y_test = k_folds
        y_train = y_train.flatten().astype(int)
        y_test = y_test.flatten().astype(int)
        f1_score = train(model=model, x_train=x_train, y_train=y_train,x_test=x_test, y_test=y_test, verbose=verbose, save=False)
        f1_score_sum += f1_score
        if f1_score > best_score:
            best_model = model
            best_score = f1_score
    mean_f1_score = f1_score_sum / K
    print(f"F1 score mean = {mean_f1_score}")
    best_model.f1_score = mean_f1_score
    save_model(best_model, verbose=True)
    if graphics is not None:
        graphics.add_plot(x1=best_model.loss, label_x1=best_model.file, x2=best_model.accuracy, label_x2=best_model.file)

def predict(data, file_model, verbose, split, graphics):
    """
    make prediction with the file_model
    """
    _, _, x_test, y_test = prepare_data(data, verbose, split)
    model = None
    with open(file_model, "rb") as f:
        model = pickle.load(f)
    print(f"Prediction on datatest ({len(x_test)} lines)")  
    out = model.predict(x_test)
    print_succes(out, y_test, model, graphics)

def best(verbose=False, graphics=None):
    """
    return the best Model
    """
    tab_models = load_models('models/neural_network_params.yml')
    best_f1 = 0
    best_model_f1 = None
    for model in tab_models:
        if verbose:
            print(model)
        f1 = model.get_accuracy()
        if f1 > best_f1:
            best_f1 = f1
            best_model_f1 = model
    if graphics is not None and best_model_f1 is not None:
        graphics.add_plot(x1=best_model_f1.loss, label_x1=best_model_f1.file, x2=best_model_f1.accuracy, label_x2=best_model_f1.file)
    if best_model_f1 is not None:
        if verbose:
            print(f"Best Model is {colors.green}{best_model_f1.file}{colors.reset} with f1 score = {colors.blue}{best_model_f1.f1_score}{colors.reset}")
        return f"models/{best_model_f1.file}"
    else:
        if verbose:
            print(f"Models {colors.red}Not Found{colors.reset}")
        return None

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:m:o:s:k:vhtpgb", ["file=", "predict", "train", "model=", "help", "split=", "verbose", "graphics", "best","K="])
    except getopt.GetoptError as inst:
        usage(inst)

    try:
        graphics = None
        mode = None
        data = None
        verbose = False
        model = None
        split = 0.8
        K = 10
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                data = load_data(arg, header=None)
        if data is None:
            usage("No data", extend=True)
        for opt, arg in opts:
            if opt in ["-t", "--train"]:
                mode = "train"
            elif opt in ["-p", "--predict"]:
                mode = "predict"
            elif opt in ["-b", "--best"]:
                mode = "find best"
            elif opt in ["-v", "--verbose"]:
                verbose = True
            elif opt in ["-o", "--model"]:
                model = arg
            elif opt in ["-h", "--help"]:
                usage(extend=True)
            elif opt in ["-g", "--graphics"]:
                graphics = graph(mode)
            elif opt in ["-s", "--split"]:
                split = float(arg)
                if split < 0 or split >=1:
                    usage("The option --split must be a float >= 0 and < 1")
            elif opt in ["-k", "--K"]:
                K = int(arg)
        if mode not in ["train", "predict", "find best"]:
            usage("Bad Mode")
        print(f"********** {colors.green}{mode.upper()}{colors.reset} **********")
        if mode == "train":
            if model is not None:
                model = get_model('models/neural_network_params.yml', model_name=model)
                if mode is not None:
                    loop_train_cross_data(data=data, K=K, model_empty=model, verbose=verbose, graphics=graphics)
            else:
                loop_multi_training(data=data, K=K, verbose=verbose, graphics=graphics)
        elif mode == "find best":
            best(verbose=verbose, graphics=graphics)
        else: # Predict mode
            if model is None:
                model = best(verbose=verbose)
                if model is None:
                    usage("no model has been trained")
            predict(data=data,file_model=model, verbose=verbose, split=split, graphics=graphics)
        if graphics is not None:
            graphics.show()
        
    except Exception as inst:
        usage(inst)

if __name__ == "__main__":
    main(sys.argv[1:])
    print("Good by !")