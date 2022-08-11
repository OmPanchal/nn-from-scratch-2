import numpy as np
from nnv3.activations.Tanh import Tanh
from nnv3.layers.Dense import Dense
from nnv3.layers.Input import Input
from nnv3.models.Model import Model
from nnv3.losses.MSE import MSE
from nnv3.optimizers.Momentum import Momentum
from nnv3.optimizers.RMSProp import RMSProp
from nnv3.optimizers.SGD import SGD


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])[..., np.newaxis]

model = Model([
	Input(2, dtype=np.float64),
	Dense(10, Tanh()),
	Dense(1)
])

model.build(optimizer=Momentum(),
			loss=MSE())

model.fit(X, Y, epochs=1000, batch_size=1)

for i in range(len(X)):
	print(X[i], model.predict(X[i]))