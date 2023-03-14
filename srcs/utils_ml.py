import numpy as np
import math
from .common import colors

def data_spliter(x: np.ndarray, y: np.ndarray, proportion: float=0.8):
    """
    split data into a train set and a test set, respecting to the given proportion
    return (x_train, x_test, y_train, y_test)
    """
    np.random.seed(42)
    if (not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(proportion, float)):
        print("spliter invalid type")
        return None
    if (x.shape[0] != y.shape[0]):
        print("spliter invalid shape")
        return None
    arr = np.concatenate((x, y), axis=1)
    N = len(y)
    X = arr[:, :x.shape[1]]
    Y = arr[:, x.shape[1]]
    sample = int(proportion*N)
    np.random.shuffle(arr)
    x_train, x_test, y_train, y_test = np.array(X[:sample, :], dtype=np.float64), np.array(X[sample:, :], dtype=np.float64), np.array(Y[:sample, ]).reshape(-1, 1), np.array(Y[sample:, ]).reshape(-1, 1)
    return (x_train, y_train, x_test, y_test)


def batch(x: np.ndarray, y: np.ndarray, m: int=32):
    """
    divide array x and y in many sub array of size m
    """
    try:
        arr = np.concatenate((x, y), axis=1)
        N = len(y)
        np.random.shuffle(arr)
        X = arr[:, :x.shape[1]]
        Y = arr[:, x.shape[1]].reshape(len(y), 1)
        batch_x = np.array_split(X, math.ceil(N / m), axis=0)
        batch_y = np.array_split(Y, math.ceil(N / m), axis=0)
        return batch_x, batch_y
    except Exception as inst:
        raise inst

def cross_validation(x, y, K):
    """
    split data into n parts
    """
    if (not isinstance(x, np.ndarray) or not isinstance(x, np.ndarray)):
        print("spliter invalid type")
        return None
    if (x.shape[0] != y.shape[0]):
        print("spliter invalid shape")
        return None
    arr = np.concatenate((x, y), axis=1)
    N = len(y)
    np.random.shuffle(arr)
    for n in range(K):
        sample = int((1 / K) * N)
        test = arr[(sample * n):(sample * (n + 1))]
        train = np.concatenate([arr[0:(sample * n)], arr[(sample * (n + 1)):N]])
        x_train, y_train, x_test, y_test = train[:, :x.shape[1]], train[:, x.shape[1]].reshape(-1, 1), test[:, :x.shape[1]], test[:, x.shape[1]].reshape(-1, 1),
        yield (x_train, y_train, x_test, y_test)


def add_polynomial_features(x, power):
    try:
        if (not isinstance(x, np.ndarray) or (not isinstance(power, int) and not isinstance(power, list))):
            print("Invalid type")
            return None
        if (isinstance(power, list) and len(power) != x.shape[1]):
            return None
        result = x.copy()
        if not isinstance(power, list):
            for po in range(2, power + 1):
                for col in x.T:
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        else:
            for col, power_el in zip(x.T, power):
                for po in range(2, power_el + 1):
                    result = np.concatenate((result, (col**po).reshape(-1, 1)), axis=1)
        return np.array(result)
    except Exception as inst:
        print(inst)
        return None

def normalize(x):
    """
    normalize matrix with minmax method
    """
    if not isinstance(x, np.ndarray):
        print("normalize Invalid type.")
        return None
    result = []
    for row in x.T:
        min_r = min(row)
        max_r = max(row)
        result.append([(el - min_r) / (max_r - min_r) for el in row])
    return np.array(result).T

def intercept_(x):
    """
    add one columns to x
    """
    try:
        if (not isinstance(x, np.ndarray)):
            print("intercept_ invalid type")
            return None
        return np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
    except Exception as inst:
        print(inst)
        return None

def category_to_bool(arr:np.ndarray):
    res = np.zeros(len(arr))
    for idx, el in enumerate(arr):
        res[idx] = np.argmax(el)
    return res

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
