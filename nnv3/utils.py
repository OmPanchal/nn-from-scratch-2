from io import TextIOWrapper
import pickle
from random import randint
import numpy as np


def load_model(filename: str):
	model_file: TextIOWrapper = open(filename, "rb")
	model = pickle.load(model_file)
	model_file.close()

	return model


def one_hot_array(arr: np.ndarray) -> np.ndarray:

    assert len(arr.shape) == 1

    classes: np.ndarray = np.unique(arr)
    output: list = []
    arr: np.ndarray = np.squeeze(arr)

    for i in arr:
        array: np.ndarray = np.zeros(shape=(classes.size))
        for idx, _class in enumerate(classes):
            if i == _class: array[idx] = 1
            else: pass
        
        output.append(array)

    return np.array(output)
