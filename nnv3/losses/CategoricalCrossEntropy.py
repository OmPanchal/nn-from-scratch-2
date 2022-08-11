import numpy as np
from nnv3.losses.Loss import Loss


class CategoricalCrossEntropy(Loss):
	def __init__(self) -> None:
		super().__init__(self.categorical_cross_entropy, self.categorical_cross_entropy_prime)

	def categorical_cross_entropy(self, pred: np.ndarray, act: np.ndarray) -> np.ndarray:
		return np.nan_to_num(-(np.sum(act * np.log(pred + 1e-8))))

	def categorical_cross_entropy_prime(self, pred: np.ndarray, act: np.ndarray) -> np.ndarray:
		return np.nan_to_num(pred - act)