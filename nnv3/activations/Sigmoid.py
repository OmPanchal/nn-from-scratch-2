import numpy as np
from nn.activations.Activation import Activation


class Sigmoid(Activation):
	def __init__(self):
		activation = self.sigmoid
		activation_prime = self.sigmoid_prime
		super().__init__(activation, activation_prime)

	def sigmoid(self, x: np.ndarray) -> np.ndarray:
		return np.nan_to_num(1 / (1 + np.exp(-x)))

	def sigmoid_prime(self, x: np.ndarray) -> np.ndarray:
		return np.nan_to_num(self.sigmoid(x) * (1 - self.sigmoid(x)))