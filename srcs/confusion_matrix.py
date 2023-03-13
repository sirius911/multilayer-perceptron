import numpy as np
import pandas as pd


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
        df_option: optional, This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """

    if labels is None:
        true_labels = list(dict.fromkeys(y_true.flatten()))
        predicted_labels = list(dict.fromkeys(y_hat.flatten()))
        if len(true_labels) > len(predicted_labels):
            labels = true_labels
        else:
            labels = predicted_labels\
    
    labels.sort()   # the labels may not be put in the same order between y and y_hat 
    dim = (len(labels), len(labels))
    result = np.zeros(dim)

    for i, (true, pred) in enumerate(zip(y_true, y_hat)):
        if true not in labels or pred not in labels:
            continue
        tl = labels.index(true) #true label
        pl = labels.index(pred) #pred label
        result[tl][pl] += 1

    if df_option:
        return pd.DataFrame(result, index=labels, columns=labels, dtype=int)
    return result.astype(np.int64)