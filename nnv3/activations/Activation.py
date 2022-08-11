import numpy as np


class Activation(object):
	def __init__(self, activation, activation_prime) -> None:
		self.input = None
		self.activation = activation
		self.activation_prime = activation_prime

	def __str__(self) -> str: return type(self).__name__

	def forward(self, input: np.ndarray) -> np.ndarray:
		self.input: np.ndarray = input
		return self.activation(input)

	def backward(self, error: np.ndarray) -> np.ndarray:
		return np.multiply(error, self.activation_prime(self.input))
