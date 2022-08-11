import numpy as np
from nnv3.optimizers.Optimizer import Optimizer


class Momentum(Optimizer):
	def __init__(self, learning_rate: float = 0.01, beta: float = 0.9) -> None:
		super().__init__(learning_rate)
		self.beta: float = beta
		self.prev_weights_grad: np.ndarray = 0
		self.prev_bias_grad: np.ndarray = 0

	def __call__(self, *args: object, **kwargs: object) -> object:
		return type(self)(learning_rate=self.learning_rate,
						beta=self.beta) 

	def __momentum(self, grad: np.ndarray, prev_grad: np.ndarray) -> np.ndarray:
		return (self.beta * prev_grad) + (self.learning_rate * grad)

	def call(self, weights_grad: np.ndarray, biases_grad: np.ndarray) -> tuple:
		self.prev_weights_grad = self.__momentum(weights_grad, self.prev_weights_grad)
		self.prev_bias_grad = self.__momentum(biases_grad, self.prev_bias_grad)

		return (self.prev_weights_grad, self.prev_bias_grad)