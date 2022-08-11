# nn-from-scratch-2

**Neural Network library only using [numpy](https://numpy.org/).** (inspired by keras) 

* **INFO**: For any of the examples which require the mnist dataset, please download the dataset from **[here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).**
* **INFO**: This is an upgraded version of a **[previous project](https://github.com/OmPanchal/nn-from-scratch)**

After you have prepared your data...
1. **Design your model using**
```python
# example prepared data
X = ... # add the features here (numpy array)
Y = ... # add the labels here (numpy array)

# Design the model
model = Model([
  Input(1, dtype="float32"),
  Dense(128, activation="tanh", use_bias=False),
  Dense(64, activation="relu", bias_initializer="uniform")
])
```
2. **Build your model (using `model.build`) and specify the optimizer and loss function**
```python
# build the model
model.build(
  optimizer="adam",
  loss="mse"
)
```
3. **Train your model (using `model.fit`)**
```python
# train the model
model.fit(
  X,
  Y,
  epochs=100,
  batch_size=16
)
```
4. **Make predictions with your model (using `model.predict`)**
```python
# predict using the model
model.predict(test_data)
```
