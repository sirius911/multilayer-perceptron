import getopt, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from common import load_data, error

def scatterplot(d: pd.DataFrame, begin : int, end : int):
    data = d.drop(d.columns[[0]], axis=1, inplace=False)
    data.iloc[:,0] = data.iloc[:,0].replace({'M': 'Malignant', 'B': 'Benign'})
    r = [1]
    for i in range(begin, end+1):
        r.append(i)
    if len(r) <= 10:
        g = sns.pairplot(data[r], hue=1, dropna=True, palette={'Malignant': 'red', 'Benign':'green'})
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
            print(f"Creating pairplot from features #{rr[1]} to #{rr[-1]}...", end='')
            g = sns.pairplot(data[rr], hue=1, dropna=True, palette={'Malignant': 'red', 'Benign':'green'})
            g._legend.set_title("Diagnosis")
            i = j
            j = j + pas
            if j > len(r):
                j = len(r)
            print("Ok")

    plt.show()

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:b:e:", ["file=", "begin=", "end="])
    except getopt.GetoptError as inst:
        error(inst)
    try:
        data = None
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                data = load_data(arg, None)
        if data is None:
            error("Data must be specified...")
        begin, end = 1, data.shape[1] - 2
        for opt, arg in opts:
            # print(f"opt = {opt} et arg = {arg}")
            if opt in ["-b", "--begin"]:
                begin = int(arg)
            elif opt in ["-e", "--end"]:
                end = int(arg)
        if begin > 0 and end <= data.shape[1] - 2:
            scatterplot(data, begin+1, end)
        else:
            error(f"Begin must be > 0 and end <= {data.shape[1] - 2}")
    except Exception as inst:
        error(inst)

if __name__ == "__main__":
    main(sys.argv[1:])