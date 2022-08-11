import numpy as np


class Optimizer(object):
	def __init__(self, learning_rate) -> None:
		self.learning_rate = learning_rate

	def call(self, *args, **kwargs): ...

	def __call__(self, *args: object, **kwargs: object) -> object: ...

	def __str__(self) -> str: return type(self).__name__
 
 