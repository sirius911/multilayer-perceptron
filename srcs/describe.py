import numpy as np
import pandas as pd

from .statistician import Statistician
from .common import error


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