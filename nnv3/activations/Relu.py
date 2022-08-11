import numpy as np
from nnv3.activations.Activation import Activation


class Relu(Activation):
	def __init__(self) -> None:
		activation = self.activation
		activation_prime = self.activation_prime
		super().__init__(activation, activation_prime)

	def activation(self, x: np.ndarray) -> np.ndarray:
		return np.maximum(0, x)

	def activation_prime(self, x: np.ndarray) -> np.ndarray:
		return x > 1