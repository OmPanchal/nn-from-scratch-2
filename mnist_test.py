import numpy as np
import pandas as pd

from nnv3.models.Model import Model
from nnv3.layers.Dense import Dense
from nnv3.losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from nnv3.layers.Input import Input
from nnv3.layers.Softmax import Softmax
from nnv3.utils import one_hot_array

# Preprocessing the data
df = pd.read_csv("dataset\mnist_train.csv")

numpydf = df.to_numpy()

batch = numpydf.T[1:].T[0:100]
labels = numpydf.T[0].T[0:100]

# change the labels to one hot arrays, to match the network output
one_hot_labels = one_hot_array(labels)

# Build the model
layers = [
	Input(784, dtype="float64"),
	Dense(64, activation="relu", bias_initializer="uniform"),
	Dense(128, activation="tanh", bias_initializer="uniform"),
	Dense(10, bias_initializer="uniform"), Softmax()]

model = Model(layers)

# Make sure to build the model
model.build(
	optimizer="adam",
	loss=CategoricalCrossEntropy()
)

# Train the model 
model.fit(
	batch,
	one_hot_labels,
	epochs=100,
    batch_size=1,
    shuffle=False
)

# Plot the loss of the model
model.graph()

# Save the model as a pickle file 
model.save("model")

# check accuracy with the first 100 examples
for i in range(100):
    predict = model.predict(batch[i])
    print("-----------------------------------------" if np.argmax(predict) != labels[i] else "")
    print(f"ACTUAL: {labels[i]}, PREDICTED: {np.argmax(predict)}")
    print("-----------------------------------------" if np.argmax(predict) != labels[i] else "")



