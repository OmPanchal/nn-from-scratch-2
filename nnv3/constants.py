import os
from re import L
import numpy as np
from nn.activations.Relu import Relu
from nnv3.activations.Tanh import Tanh
from nnv3.functions import normal, uniform
from nnv3.activations.Sigmoid import Sigmoid
from nnv3.losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from nnv3.losses.MSE import MSE
from nnv3.optimizers.Adam import Adam
from nnv3.optimizers.Momentum import Momentum
from nnv3.optimizers.RMSProp import RMSProp
from nnv3.optimizers.SGD import SGD


SYSTEM_ARCHITECTURE: int = 64
if os.environ.get('PROCESSOR_ARCHITECTURE').lower() == 'x86' and os.environ.get('PROCESSOR_ARCHITEW6432') is None:
	SYSTEM_ARCHITECTURE: int = 32


INITIALIZERS = {
	"ones": np.ones,
	"zeros": np.zeros,
	"uniform": uniform,
	"normal": normal
}

ACTIVATIONS = {
	"Tanh": Tanh,
	"tanh": Tanh,
	"Relu": Relu,
	"relu": Relu,
	"Sigmoid": Sigmoid,
	"sigmoid": Sigmoid
}

OPTIMIZERS = {
	"SGD": SGD,
	"sgd": SGD,
	"Momentum": Momentum,
	"momentum": Momentum,
	"RMSProp": RMSProp,
	"rmsprop": RMSProp,
	"Adam": Adam,
	"adam": Adam
}

LOSSES = {
	"MSE": MSE,
	"mse": MSE,
	"CategoricalCrossEntropy": CategoricalCrossEntropy,
	"categoricalerossentropy": CategoricalCrossEntropy,
	"CCE": CategoricalCrossEntropy
}