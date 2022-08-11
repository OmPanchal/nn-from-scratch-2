


import numpy as np
from nnv3.optimizers.Optimizer import Optimizer


class Adam(Optimizer):
	def __init__(self, learning_rate: float = 0.001,
				 beta1: float = 0.9, 
				 beta2: float = 0.999, 
				 epsilon: float = 1e-8) -> None:
		super().__init__(learning_rate)
		self.beta1: float = beta1
		self.beta2: float = beta2
		self.epsilon: float = epsilon

		self.prev_weightsM_grad: np.ndarray = 0
		self.prev_biasM_grad: np.ndarray = 0

		self.prev_weightsR_grad: np.ndarray = 0
		self.prev_biasR_grad: np.ndarray = 0

	def __call__(self, *args: object, **kwargs: object) -> object:
		return type(self)(self.learning_rate,
						  self.beta1,
						  self.beta2,
						  self.epsilon)

	def __momentum(self, grad: np.ndarray, prev_grad: np.ndarray) -> np.ndarray:
		return (self.beta1 * prev_grad) + ((1 - self.beta1) * grad)

	def __rmsprop(self, grad: np.ndarray, prev_grad: np.ndarray) -> np.ndarray:
		return (self.beta2 * prev_grad) + ((1 - self.beta2) * (grad ** 2))

	def __root(self, M: np.ndarray, R: np.ndarray) -> np.ndarray:
		return ((M) / np.sqrt(R + self.epsilon)) * self.learning_rate

	def call(self, weights_grad: np.ndarray, bias_grad: np.ndarray) -> np.ndarray:
		self.prev_weightsM_grad = self.__momentum(weights_grad, self.prev_weightsM_grad)
		self.prev_biasM_grad = self.__momentum(bias_grad, self.prev_biasM_grad)

		self.prev_weightsR_grad = self.__rmsprop(weights_grad, self.prev_weightsR_grad)
		self.prev_biasR_grad = self.__rmsprop(bias_grad, self.prev_biasR_grad)

		return (self.__root(self.prev_weightsM_grad, self.prev_weightsR_grad),
				self.__root(self.prev_biasM_grad, self.prev_biasR_grad))


# ~ second working method

# import numpy as np
# from nnv3.optimizers.Optimizer import Optimizer
# from nnv3.optimizers.Momentum import Momentum
# from nnv3.optimizers.RMSProp import RMSProp


# class Adam(Optimizer):
# 	def __init__(self, learning_rate: float = 0.001,
# 				 beta1: float = 0.9,
# 				 beta2: float = 0.999,
# 				 epsilon: float = 1e-8) -> None:

# 		super().__init__(learning_rate)
# 		self.beta1: float = beta1
# 		self.beta2: float = beta2
# 		self.epsilon: float = epsilon

# 		self.M: Momentum = Momentum(beta=self.beta1)
# 		self.R: RMSProp = RMSProp(beta=self.beta2, epsilon=self.epsilon)

# 		self.t = 0

# 	def __call__(self, *args: object, **kwargs: object) -> object:
# 		return type(self)(self.learning_rate,
# 						  self.beta1,
# 						  self.beta2,
# 						  self.epsilon)
						  
# 	def call(self, weights_grad: np.ndarray, bias_grad: np.ndarray):
# 		MW, MB = self.M.call(weights_grad, bias_grad)
# 		RM, RB = self.R.call(MW, MB)

# 		return RM, RB