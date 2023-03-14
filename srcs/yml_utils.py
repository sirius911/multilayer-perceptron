import os
import pickle
import yaml
from .common import error
from .network import Network
from .layer import DenseLayer

def load_params(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            config = yaml.safe_load(file)
        return config
    else:
        error(f"{file_name} does not exist.")

def create_models(file_params = 'models/neural_network_params.yml'):
    """
        Function loading 'neural_network_params' file
        and return a list of model Network with the params of the file
    """
    config = load_params(file_name=file_params)
    ret_models = []
    for model_config in config['networks']:
        model = Network()
        model.file = model_config['name']
        model.epochs = model_config['iter']
        for layer in model_config['layers']:
            model.add(DenseLayer(layer['neurons'], layer['activation']))
        ret_models.append(model)
    return ret_models

def load_models(file_params = 'models/neural_network_params.yml'):
    """
    return a list of trained models
    """
    config = load_params(file_name=file_params)
    ret_models = []
    for model_config in config['networks']:
        file_name = f"models/{model_config['name']}"
        with open(file_name, "rb") as f:
            model = pickle.load(f)
        ret_models.append(model)
    return ret_models

def get_model(file_params = 'models/neural_network_params.yml', model_name='models/model.pkl'):
    """
    return a Model Network named model_name in the file_params yml file
    """
    config = load_params(file_name=file_params)
    for model_config in config['networks']:
        if model_config['name'] == model_name:
            model = Network()
            model.file = model_config['name']
            model.epochs = model_config['iter']
            for layer in model_config['layers']:
                model.add(DenseLayer(layer['neurons'], layer['activation']))
            return model
    error(f"{model_name} does not exist in '{file_params}'")
