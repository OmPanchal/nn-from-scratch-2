from nnv3.activations.Activation import Activation
from nnv3.constants import ACTIVATIONS, INITIALIZERS
from nnv3.functions import display_layer
from nnv3.layers.Layer import Layer
from nnv3.optimizers.Optimizer import Optimizer

import numpy as np


class Dense(Layer):

	def __init__(self, units: int,
				 activation: Activation = None,
				 use_bias: bool = True,
				 weights_initializer: str = "uniform",
				 bias_initializer: str = "zeros"
				 ) -> None:

		super().__init__(units or None)
		self.prev_units: int = None

		if isinstance(activation, Activation):
			self.activation: Activation = activation 
		elif type(activation) == str: self.activation = ACTIVATIONS[activation]()
		elif  activation is None: self.activation = None
		else: raise TypeError(f"{type(activation)} is an invalid type for activation")

		self.use_bias: bool = use_bias 

		self.weight_grad: np.ndarray = 0
		self.bias_grad: np.ndarray = 0

		self.weights_initializer: str = weights_initializer
		self.bias_initializer: str = bias_initializer

		self.optimizer: Optimizer = None
		self.dtype: str = None
	
	def __call__(self, prev_layer: Layer, *args: object, **kwargs: object) -> object:
		self.prev_units = prev_layer.units
		self.dtype = prev_layer.dtype

		if "optimizer" in kwargs.keys():
			self.optimizer = kwargs.get("optimizer")

		self.weights: np.ndarray = INITIALIZERS[self.weights_initializer]((self.units, self.prev_units)).astype(self.dtype)
		if self.use_bias: 
			self.biases: np.ndarray = INITIALIZERS[self.bias_initializer]((self.units, 1)).astype(self.dtype)
		else: self.biases = 0

		return super().__call__(*args, **kwargs)

	def __str__(self, **kwargs) -> str:
		return display_layer(self,
			activation=self.activation,
			units=self.units,
			dtype=self.dtype,
			optimizer=self.optimizer)

	def forward(self, input: np.ndarray) -> np.ndarray:
		self.input: np.ndarray = input
		output: np.ndarray = np.dot(self.weights, self.input)

		if self.use_bias: output += self.biases

		if self.activation: return self.activation.forward(output)
		else: return output

	def backward(self, error: np.ndarray) -> np.ndarray:
		if self.activation: activation_error = self.activation.backward(error)
		else: activation_error = error

		self.weight_grad += np.dot(activation_error, self.input.T)
		if self.use_bias: self.bias_grad += activation_error
		else: self.bias_grad = 0

		return np.dot(self.weights.T, error)

	def update(self, **kwargs) -> None:
		if "batch_size" in kwargs.keys():
			batch_size = kwargs.get("batch_size")

		self.weight_grad /= batch_size
		self.bias_grad /= batch_size

		bias_grad = self.bias_grad if self.use_bias else 0 

		weights_update, bias_update = self.optimizer.call(self.weight_grad, bias_grad)
		self.weights -= weights_update
		self.biases -= bias_update
		
		self.weight_grad = 0
		self.bias_grad = 0