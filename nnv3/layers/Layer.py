import numpy as np


class Layer(object):
	def __init__(self, units: int) -> None:
		self.units: int = units

	def forward(self, input: np.ndarray) -> np.ndarray:...

	def backward(self, error: np.ndarray) -> np.ndarray: ...

	def update(self, **kwargs) -> None: ...

	def __call__(self, *args: object, **kwargs: object) -> object:
		return self

	def __str__(self) -> str: ...