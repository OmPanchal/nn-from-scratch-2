import numpy as np
from nnv3.optimizers.Optimizer import Optimizer


class SGD(Optimizer):
	def __init__(self, learning_rate: int = 0.1) -> None:
		super().__init__(learning_rate)

	def __call__(self, *args: object, **kwargs: object) -> object:
		return type(self)(learning_rate=self.learning_rate)

	def call(self, weights_grad: np.ndarray, biases_grad: np.ndarray) -> tuple:
		weights_update: np.ndarray = self.learning_rate * weights_grad
		bias_update: np.ndarray = self.learning_rate * biases_grad

		return (weights_update, bias_update)

