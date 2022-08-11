import numpy as np
from nnv3.optimizers.Optimizer import Optimizer


class RMSProp(Optimizer):
	def __init__(self, learning_rate: float = 0.001,
				 beta: float = 0.999,
				 epsilon: float = 1e-8) -> None:

		super().__init__(learning_rate)
		self.beta: float = beta
		self.epsilon: float = epsilon

		self.prev_weights_grad: np.ndarray = 0
		self.prev_bias_grad: np.ndarray = 0

	def __call__(self, *args: object, **kwargs: object) -> object:
		return type(self)(learning_rate=self.learning_rate,
						 beta=self.beta,
						 epsilon=self.epsilon)
					
	def __rmsprop(self, grad: np.ndarray, prev_grad: np.ndarray) -> np.ndarray:
		return (self.beta * prev_grad) + ((1 - self.beta) * (grad ** 2))

	def __root(self, grad: np.ndarray, prev_grad: np.ndarray) -> np.ndarray:
		return ((grad) / np.sqrt(prev_grad + self.epsilon)) * self.learning_rate

	def call(self, weights_grad: np.ndarray, bias_grad: np.ndarray) -> np.ndarray:
		self.prev_weights_grad = self.__rmsprop(weights_grad, self.prev_weights_grad)
		self.prev_bias_grad = self.__rmsprop(bias_grad, self.prev_bias_grad)

		return (self.__root(weights_grad, self.prev_weights_grad),
				self.__root(bias_grad, self.prev_bias_grad))

