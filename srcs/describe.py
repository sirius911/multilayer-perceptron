import numpy as np
import getopt, sys
import pandas as pd

from statistician import Statistician
from common import load_data, error


def get_description(data: pd.DataFrame, labelName: bool):
    res = {}
    for idx, col_name in enumerate(data.columns[1:]):
        name = idx
        if labelName:
            name = col_name
        
        number_nan = Statistician().count_nan(data[col_name].to_numpy())
        
        value = np.array(data[col_name].dropna())
        quartile = Statistician().quartile(value)
        res[name] = [
            Statistician().count(value),
            Statistician().mean(value),
            Statistician().std(value),
            Statistician().min(value),
            quartile[0] if quartile is not None else None,
            Statistician().median(value),
            quartile[1] if quartile is not None else None,
            Statistician().max(value),
            Statistician().var(value),
            number_nan,
        ]
    return res


def describe(data: pd.DataFrame, begin: int, end: int, labelName: bool):
    if end < begin or end < 0 or begin < 0 or end > data.shape[1]:
        error("Invalid end or begin argument")
    data = get_description(data, labelName)
    df = pd.DataFrame(data, index=["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Var", "NaN"])
    print(df.iloc[:, begin:end])


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "f:b:e:n", ["file=", "begin=", "end=", "name"])
    except getopt.GetoptError as inst:
        error(inst)

    try:
        for opt, arg in opts:
            if opt in ["-f", "--file"]:
                data = load_data(arg)
        begin, end, labelName = 0, data.shape[1], False
        for opt, arg in opts:
            if opt in ["-b", "--begin"]:
                begin = int(arg)
            elif opt in ["-e", "--end"]:
                end = int(arg)
            elif opt in ["-n", "--name"]:
                labelName = True
        describe(data, begin, end, labelName)
    except Exception as inst:
        error(inst)


if __name__ == "__main__":
    main(sys.argv[1:])