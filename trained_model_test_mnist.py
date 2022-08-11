from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from nnv3.utils import load_model, one_hot_array


model = load_model("saved_models\model.pickle")

df = pd.read_csv("dataset\mnist_test.csv")

batch = df.to_numpy().T[1:].T[0:]
labels = df.to_numpy().T[0].T[0:]
one_hot_labels = one_hot_array(labels)

running = True

while running:
	randidx = input("Input an index between 0 and 9999 (both inclusive): ")
	if randidx == "exit":
		running = False
		continue
	else: randidx = int(randidx)

	input_value = np.reshape(batch[randidx], newshape=(28, 28))
	predicted_values = model.predict(batch[randidx])
	argmax_predicted = np.argmax(predicted_values)
	classes = np.unique(labels)

	print(np.squeeze(predicted_values), np.squeeze(one_hot_labels[randidx]))

	fig, axes = plt.subplots(nrows=2, ncols=1)

	# change the colour depending on the predicted output
	colour = np.repeat("b", predicted_values.size)

	if argmax_predicted == labels[randidx]: colour[argmax_predicted] = "g"
	if argmax_predicted != labels[randidx]: 
			colour[argmax_predicted] = "r"
			colour[labels[randidx]] = "g"

	# plot the data
	axes[0].imshow(np.squeeze(input_value))
	axes[1].bar(classes, np.squeeze(predicted_values), color=colour)

	plt.xticks(classes)
	plt.show()
