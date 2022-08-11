import numpy as np
from nnv3.activations.Activation import Activation


class Tanh(Activation):
	def __init__(self) -> None:
		activation = self.activation
		activation_prime = self.activation_prime
		super().__init__(activation, activation_prime)

	def activation(self, x: np.ndarray) -> np.ndarray: return np.tanh(x)
	def activation_prime(self, x: np.ndarray) -> np.ndarray: return 1 - np.tanh(x) ** 2

