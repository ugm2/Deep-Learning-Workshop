from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Set random seed for reproducibility
np.random.seed(1)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Print shapes
print("Number of training examples: m_train = " + str(len(x_train)))
print("Number of testing examples: m_test = " + str(len(x_test)))
print("Height/Width of each image: num_px = " + str(x_train.shape[1]))
print(f"x_train.shape: {x_train.shape}")
print(f"x_test.shape: {x_test.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"y_test.shape: {y_test.shape}")

# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# Convert labels to categorical
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Create the model
model = keras.Sequential(
    [
        layers.Input(shape=(784,)),  # Input layer (flattened 28x28 images)
        layers.Dense(
            128, activation="relu", kernel_initializer="glorot_uniform"
        ),  # Hidden layer 1
        layers.Dense(
            64, activation="relu", kernel_initializer="glorot_uniform"
        ),  # Hidden layer 2
        layers.Dense(
            32, activation="relu", kernel_initializer="glorot_uniform"
        ),  # Hidden layer 3
        layers.Dense(
            10, activation="softmax", kernel_initializer="glorot_uniform"
        ),  # Output layer
    ]
)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.02),
    loss="categorical_crossentropy",
    metrics=["accuracy", "Precision", "Recall"],
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=1
)

# Evaluate the model
print("\nTraining data:")
train_results = model.evaluate(x_train, y_train, verbose=0)
print(f"Loss: {train_results[0]:.2f}")
print(f"Accuracy: {train_results[1]:.2f}")
print(f"Precision: {train_results[2]:.2f}")
print(f"Recall: {train_results[3]:.2f}")

print("\nValidation data:")
test_results = model.evaluate(x_test, y_test, verbose=0)
print(f"Loss: {test_results[0]:.2f}")
print(f"Accuracy: {test_results[1]:.2f}")
print(f"Precision: {test_results[2]:.2f}")
print(f"Recall: {test_results[3]:.2f}")
