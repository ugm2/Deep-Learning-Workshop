from dl_workshop.logistic_regression import LogisticRegression
import numpy as np
import h5py
import os
import git

np.random.seed(1)

repo = git.Repo(".", search_parent_directories=True)
git_root = repo.working_tree_dir


def load_data():
    train_dataset = h5py.File(
        os.path.join(
            git_root, "examples/BinaryClassification/data/train_catvnoncat.h5"
        ),
        "r",
    )
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File(
        os.path.join(git_root, "examples/BinaryClassification/data/test_catvnoncat.h5"),
        "r",
    )
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x_orig.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x_orig.shape))
print("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(
    train_x_orig.shape[0], -1
).T  # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.0
test_x = test_x_flatten / 255.0

print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))

lr_model = LogisticRegression(learning_rate=0.0075, verbose=False)
lr_model.fit(train_x, train_y, epochs=400, validation_data=(test_x, test_y))

# Evaluate model with more metrics
print("Training data:")
print(lr_model.evaluate(train_x, train_y))
print("Validation data:")
print(lr_model.evaluate(test_x, test_y))
