import numpy as np
from nnv3.losses.Loss import Loss

class MSE(Loss):
	def __init__(self) -> None:
		super().__init__(self.mse, self.mse_prime)

	def mse(self, pred: np.ndarray, act: np.ndarray) -> np.ndarray:
		return (2 / len(act)) * np.sum((act - pred) ** 2)

	def mse_prime(self, pred: np.ndarray, act: np.ndarray) -> np.ndarray:
		return 2 * (pred - act) / np.size(act)