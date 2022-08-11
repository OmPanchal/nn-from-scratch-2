import numpy as np
from nnv3.layers.Layer import Layer


class Softmax(Layer):
	def __init__(self) -> None:
		units = None
		self.dtype: str = None
		self.output: np.ndarray = None
		super().__init__(units)

	def __call__(self, prev_layer: Layer, *args: object, **kwargs: object) -> object:
		self.dtype = prev_layer.dtype

	def forward(self, input: np.ndarray) -> np.ndarray:
		input: np.ndarray = input.astype(self.dtype)
		tmp: np.ndarray = np.exp(input) + np.max(input)
		self.output: np.ndarray = tmp / np.sum(tmp)

		return np.nan_to_num(self.output)

	def backward(self, error: np.ndarray) -> np.ndarray:
		error: np.ndarray = error.astype(self.dtype)
		n: int = np.size(self.output)
		tmp: np.ndarray = np.tile(self.output, n)
	
		return np.nan_to_num(np.dot(tmp * (np.identity(n) - tmp.T), error)) 

	def update(self, **kwargs) -> None:
		return super().update(**kwargs)