import numpy as np
import pandas as pd
import sys
import yaml


_NUMERIC_KINDS = set('buifc')


class colors:
    green = '\033[92m' # vert
    blue = '\033[94m' # blue
    yellow = '\033[93m' # jaune
    red = '\033[91m' # rouge
    reset = '\033[0m' #gris, couleur normales

def error(msg: str="", exit: int=2, color: str=colors.red):
    print(f"{color}{msg}")
    sys.exit(exit)

def load_data(path: str, header : int = 0, names : list = None):
    try:
        with open(path, "r") as stream:
            data = pd.read_csv(stream, header=header, names=names)
    except Exception as inst:
        error(inst)
    return data

def error(msg: str="", exit: int=2, color: str=colors.red):
    print(f"{color}{msg}")
    sys.exit(exit)

def load_yml_file(path: str):
    try:
         with open(path, "r") as stream:
            data = yaml.safe_load(stream)
    except Exception as inst:
        error(inst)
    return data


def is_numeric(array: np.ndarray):
    """Determine whether the argument has a numeric datatype, when
    converted to a NumPy array.
    """
    return np.asarray(array).dtype.kind in _NUMERIC_KINDS
