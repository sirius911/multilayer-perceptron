import numpy as np


def perf_measure(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    TP = sum((y == pos_label) & (y_hat == pos_label))
    FP = sum((y != pos_label) & (y_hat == pos_label))
    TN = sum((y != pos_label) & (y_hat != pos_label))
    FN = sum((y == pos_label) & (y_hat != pos_label))
    return (TP, FP, TN, FN)


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp, fp, tn, fn = perf_measure(y, y_hat)
        return (tp + tn) / (tp + fp + tn + fn)
    except Exception as inst:
        raise(inst)


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp, fp, tn, fn = perf_measure(y, y_hat, pos_label)
        if tp + fp == 0:
            return 0
        return (tp) / (tp + fp)
    except Exception as inst:
        raise(inst)


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp, fp, tn, fn = perf_measure(y, y_hat, pos_label)
        if tp + fn == 0:
            return 0
        return (tp) / (tp + fn)
    except Exception as inst:
        raise(inst)


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat)
    try:
        if precision + recall == 0:
            return 0
        return float(2 * precision * recall) / (precision + recall)
    except Exception as inst:
        raise(inst)