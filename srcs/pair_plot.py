import getopt, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .common import colors

from .common import load_data, error

def scatterplot(d: pd.DataFrame, begin : int, end : int):
    # data = d.drop(d.columns[[0]], axis=1, inplace=False)
    data = d.drop('ID', axis=1, inplace=False)
    data.iloc[:,0] = data.iloc[:,0].replace({'M': 'Malignant', 'B': 'Benign'})
    column_names = data.columns
    r = [column_names[0]]

    for i in range(begin, end+1):
        r.append(column_names[i])
    if len(r) <= 10:
        print(f"Creating pairplot from column '{colors.blue}{column_names[1]}{colors.reset}' to '{colors.blue}{column_names[-1]}{colors.reset}'...", end='')
        g = sns.pairplot(data[r], hue='Diagnosis', dropna=True, palette={'Malignant': 'red', 'Benign':'green'})
        g._legend.set_title("Diagnosis")
    else:
        # div the sns pairplot by 10 features
        i = 1
        pas = 10
        j = pas
        while True:
            rr = [r[0]] + r[i:j]
            if len(rr) == 1:
                break
            print(f"Creating pairplot from column '{colors.blue}{rr[1]}{colors.reset}' to '{colors.blue}{rr[-1]}{colors.reset}'...", end='')
            g = sns.pairplot(data[rr], hue='Diagnosis', dropna=True, palette={'Malignant': 'red', 'Benign':'green'})
            g._legend.set_title("Diagnosis")
            i = j
            j = j + pas
            if j > len(r):
                j = len(r)
            print("Ok")
    plt.show()