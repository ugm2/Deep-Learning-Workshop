from tensorflow import keras
from dl_workshop.neural_network import NeuralNetwork
from dl_workshop.activation_functions import relu, softmax
from dl_workshop.cost_functions import categorical_crossentropy
from dl_workshop.parameters_initialisation import he_initialisation
from dl_workshop.regularisation_functions import l2
from tensorflow.keras.utils import to_categorical
import numpy as np

np.random.seed(1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

m_train = x_train.shape[0]
num_px = x_train.shape[1]
m_test = x_test.shape[0]

# Print shapes
print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print(f"x_train.shape: {x_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")

# Reshape the training and test examples
x_train_flatten = x_train.reshape(x_train.shape[0], -1)
x_test_flatten = x_test.reshape(x_test.shape[0], -1)

# Standardize data to have feature values between 0 and 1.
x_train = x_train_flatten / 255.0
x_test = x_test_flatten / 255.0

#  shrink data size
x_train = x_train[:].T
y_train = y_train[:]
x_test = x_test[:].T
y_test = y_test[:]

# Reshape the labels from 1D to 2D
y_train = to_categorical(y_train).T
y_test = to_categorical(y_test).T

print("x_train's shape: " + str(x_train.shape))
print("x_test's shape: " + str(x_test.shape))
print("y_train's shape: " + str(y_train.shape))
print("y_test's shape: " + str(y_test.shape))

nn_model = NeuralNetwork(
    input_size=x_train.shape[0],
    layers=[(128, relu), (64, relu), (32, relu), (y_train.shape[0], softmax)],
    cost_function=categorical_crossentropy,
    initialisation_method=he_initialisation,
    regularisation_method=l2,
    lmbda=0.7,
    learning_rate=0.02,
    verbose=True,
    verbose_iteration=10,
)
nn_model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test))

# Evaluate model with more metrics
print("Training data:")
print(nn_model.evaluate(x_train, y_train))
print("Validation data:")
print(nn_model.evaluate(x_test, y_test))
