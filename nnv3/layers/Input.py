import numpy as np
from nnv3.functions import display_layer
from nnv3.layers.Layer import Layer
from nnv3.constants import SYSTEM_ARCHITECTURE


class Input(Layer):
	def __init__(self, units: int, dtype: str = f"float{SYSTEM_ARCHITECTURE}") -> None:
		self.dtype = dtype
		super().__init__(units)
		
	def __str__(self) -> str:
		return display_layer(self, units=self.units, dtype=self.dtype)

	def forward(self, input: np.ndarray) -> np.ndarray:
		return input.astype(self.dtype)

	def backward(self, error: np.ndarray) -> np.ndarray: 
		return error.astype(self.dtype)
