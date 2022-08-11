from datetime import datetime
from io import TextIOWrapper
import os
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from nnv3.constants import LOSSES, OPTIMIZERS

from nnv3.functions import split_batch
from nnv3.losses.Loss import Loss
from nnv3.layers.Layer import Layer
from nnv3.optimizers.Optimizer import Optimizer

import pickle

class Model(object):
	def __init__(self, layers: list = None) -> None:
		self.layers: list = layers if layers else None
		
		self.epochs: int = None
		self.batch_size: int = None

		self.optimizer: Optimizer = None
		self.loss: Loss = None

		self.shuffle: bool = None

		self.error_arr: list = []

	def __str__(self) -> str: ...

	def __forward(self, input: np.ndarray) -> np.ndarray:
		output: np.ndarray = input
		for layer in self.layers:
			output = layer.forward(output)

		return output 

	def __backward(self, error: np.ndarray) -> None:
		grad: np.ndarray = error

		for layer in reversed(self.layers):
			grad = layer.backward(grad)

	def __train(self, X: np.ndarray, Y: np.ndarray) -> None:
		for x, y in zip(X, Y):
			output = self.__forward(x)

			self.grad = self.loss.loss_prime(output, y)
			self.error += self.loss.loss(output, y)

			self.__backward(self.grad)

	def __update(self) -> None:
		for layer in reversed(self.layers):
			layer.update(batch_size=self.batch_size)

	def __train_batch(self, X_full: np.ndarray, Y_full: np.ndarray) -> None:
		for _x, _y in zip(X_full, Y_full):
			self.__train(_x, _y)

		self.error_arr.append(self.error)

	def add(self, layer: Layer, *args) -> None:
		self.layers.append(layer)
		if args: self.layers.append(layer for layer in args)	

	def build(self, optimizer: Optimizer , loss: Loss) -> None:
		if type(optimizer) is str:
			self.optimizer: Optimizer = OPTIMIZERS[optimizer]()
		elif isinstance(optimizer, Optimizer): self.optimizer: Optimizer = optimizer

		if type(loss) is str:
			self.loss: Loss = LOSSES[loss]()	
		elif isinstance(loss, Loss): self.loss: Loss = loss

		_: Layer = self.layers[0]()
		for layer in self.layers[1:]:
			_ = layer(_, optimizer=self.optimizer())
			
	def predict(self, input: np.ndarray) -> np.ndarray:
		input: np.ndarray = input[..., np.newaxis]
		return self.__forward(input)

	def fit(self, X: np.ndarray,
			 Y: np.ndarray, 
			 epochs: int = 1, 
			 batch_size: int = 32,
			 **kwargs) -> np.ndarray:

		self.epochs = epochs
		self.batch_size = batch_size

		X: np.ndarray = X[..., np.newaxis]
		Y: np.ndarray = Y[..., np.newaxis]

			
		(X_full, _) = split_batch(X, self.batch_size)
		(Y_full, _) = split_batch(Y, self.batch_size)

		print(f"""
		- Optimizer={self.optimizer.__str__()},
		- Loss: {self.loss.__str__()} 
		""")

		for epoch in range(self.epochs):
			self.error = 0
			self.__train_batch(X_full, Y_full)
			self.__update()
			print(f"<Epoch={epoch + 1}, Loss={self.error}>")

		return self.error_arr

	def graph(self):
		plt.xlabel("epochs")
		plt.ylabel(f"{self.loss.__str__()}")
		
		plt.plot(self.error_arr)
		plt.show()

	def save(self, filename: str,
			 destination: str = "saved_models", 
			 replace: bool = True) -> None: 

		if not os.path.exists(destination):
			os.makedirs(destination)

		if os.path.exists(os.path.join(filename, destination) + ".pickle") and replace == False:
			now: datetime = datetime.now()
			current_time = now.strftime("%D/%M/%Y--%H-%M")
			filename += current_time

		filename: str = os.path.join(destination, filename)
		filename += ".pickle"

		store: TextIOWrapper = open(filename, "wb")
		pickle.dump(self, store)
		store.close()
		