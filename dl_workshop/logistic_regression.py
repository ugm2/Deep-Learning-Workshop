import numpy as np
from dl_workshop.activation_functions import sigmoid
from dl_workshop.cost_functions import binary_crossentropy


class LogisticRegression:

    def __init__(self, learning_rate=0.01, verbose=False):
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.w, self.b = None, None
        self.grads, self.costs = None, None

    def init_params(self, dim):
        """
        Initialize weights and bias as a vector of given dimension
        """
        w = np.zeros((dim, 1))
        b = 0.0
        return w, b

    def propagate(self, X, Y):
        """
        Implement the cost function and its gradient
        """
        m = X.shape[1]

        # Forward propagation
        A = sigmoid(np.dot(self.w.T, X) + self.b)
        cost = binary_crossentropy(Y, A)

        # Backward propagation
        dw = np.dot(X, (A-Y).T) / m
        db = np.sum(A-Y) / m

        cost = np.squeeze(cost)

        grads = {"dw": dw,
                 "db": db}

        return grads, cost

    def optimize(self, X, Y, epochs):
        """
        Optimize the weights and bias by running a gradient descent algorithm
        """
        costs = []
        for i in range(epochs):
            grads, cost = self.propagate(X, Y)
            dw = grads["dw"]
            db = grads["db"]

            # Update weights
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            if i % 100 == 0:
                costs.append(cost)
                if self.verbose:
                    print("Cost after iteration %i: %f" % (i, cost))

        grads = {"dw": dw,
                 "db": db}

        return grads, costs

    def fit(self, X, Y, epochs=1, validation_data=None):
        """
        Fit the model given data
        """
        # Initialize weights and bias if None
        if self.w is None or self.b is None:
            self.w, self.b = self.init_params(X.shape[0])
        # Optimize weights and bias
        self.grads, self.costs = self.optimize(X, Y, epochs)
        # Predict the labels
        Y_prediction_train = self.predict(X)
        if validation_data is not None:
            Y_prediction_validation = self.predict(validation_data[0])
        else:
            Y_prediction_validation = None
        # Print results if verbose is True
        if self.verbose:
            print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y)) * 100))
            print("validation accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_validation - validation_data[1])) * 100))

    def predict(self, X):
        """
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        """
        w = self.w.reshape(X.shape[0], 1)

        # Compute vector "A" predicting the probabilities of the input X being of either class
        A = sigmoid(np.dot(w.T, X) + self.b)

        Y_prediction = A > np.full(A.shape, 0.5)

        return Y_prediction

    def evaluate(self, X, Y):
        """
        Evaluates the model's predictions

        Args:
            X: A numpy.ndarray with shape (nx, m) that contains the input data
            Y: A numpy.ndarray with shape (1, m) that contains the training labels
        Returns:
            accuracy: A float value containing the accuracy of the model's predictions
        """
        # Predict and apply threshold 0.5
        Y_prediction = self.predict(X)
        Y_prediction = np.where(Y_prediction > 0.5, 1, 0)
        
        # Obtain accuracy, precision, recall, and F1 score
        accuracy = np.sum(Y_prediction == Y) / Y.shape[1]
        precision = np.sum(np.logical_and(Y_prediction == Y, Y == 1)) / np.sum(Y == 1)
        recall = np.sum(np.logical_and(Y_prediction == Y, Y == 1)) / np.sum(Y == 1)
        f1 = 2 * precision * recall / (precision + recall)
        
        # Return as a dictionary
        return {'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'f1': round(f1, 2)}

    def save(self, path):
        """
        Save the model parameters to the given path
        """
        np.save(path, np.array([self.w, self.b], dtype=object))

    def load(self, path):
        """
        Load model parameters from the given path
        """
        self.w, self.b = np.load(path, allow_pickle=True)
