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

# def dataset_control(data: pd.DataFrame, type_data):
#     """ Control dataset integrity"""
#     good_columns = ['Index', 'Hogwarts House', 'First Name', 'Last Name', 'Birthday',
#        'Best Hand', 'Arithmancy', 'Astronomy', 'Herbology',
#        'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
#        'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
#        'Care of Magical Creatures', 'Charms', 'Flying']
#     nb_columns = 19
#     nb_numeric_columns = 14
#     if type_data in ['train', 'test']:
#         if len(data.columns) != nb_columns:
#             error("Bad dataset's file: the number of columns is wrong.")
#         for col in data.columns:
#             if col not in good_columns:
#                error("Bad dataset's file: Bad name in a column.")
        
#     if type_data == 'train':
#         # Hogwarts House columns must be not empty
#         if data['Hogwarts House'].isnull().sum() != 0:
#             error("Bad dataset's file: The Hogwarts House column has got empty value")
#         if len(data.select_dtypes('number').columns) != nb_numeric_columns:
#             error("Bad dataset's file: Numeric column has got non numeric value.")
#     elif type_data == 'test':
#         if data['Hogwarts House'].isnull().sum() != len(data):
#             error("Bad dataset's file: The Hogwarts House column must be empty.")
#         if len(data.select_dtypes('number').columns) != (nb_numeric_columns + 1):
#             error("Bad dataset's file: Numeric column has got non numeric value.")
#     return data

def load_data(path: str, header : int = 0):
    try:
        with open(path, "r") as stream:
            data = pd.read_csv(stream, header=header)
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
