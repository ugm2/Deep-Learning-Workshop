from sklearn.model_selection import train_test_split
from dl_workshop.neural_network import NeuralNetwork
import pandas as pd
import numpy as np
np.random.seed(1)

# Load Titanic dataset
train = pd.read_csv('data/titanic_train.csv', index_col='PassengerId')
test = pd.read_csv('data/titanic_test.csv', index_col='PassengerId')

# Fill in missing values for Age and Embarked
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# Encode categorical features Sex and Embarked
train['Sex']=train['Sex'].map({'female' : 1,'male' : 0}).astype(int)
test['Sex']=test['Sex'].map({'female' : 1,'male' : 0}).astype(int)
train = pd.concat([train, pd.get_dummies(train['Embarked'])], axis=1)
test = pd.concat([test, pd.get_dummies(test['Embarked'])], axis=1)

# Drop Cabin, Name, Ticket, Embarked
train = train.drop(columns=['Cabin', 'Name', 'Ticket', 'Embarked'])
test = test.drop(columns=['Cabin', 'Name', 'Ticket', 'Embarked'])

# Use 'Survived' column as labels
y_train = train['Survived']
X_train = train.drop(columns=['Survived'])
X_test = test
# y_test = test['Survived']
# X_test = test.drop(columns=['Survived'])

# Split train into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Dataframes to numpy arrays
X_train = X_train.values.T
X_val = X_val.values.T
y_train = y_train.values.reshape(1, y_train.shape[0])
y_val = y_val.values.reshape(1, y_val.shape[0])
X_test = X_test.values.T

nn_model = NeuralNetwork(
    layers_dict={
        'layers': [X_train.shape[0], 20, 20, 15, 7, 5, 1],
        'activations': ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid']
    },
    learning_rate=0.0075,
    verbose=False
)
nn_model.fit(X_train, y_train, epochs=15000, validation_data=(X_val, y_val))

# Evaluate model with more metrics
print("Evaluate on training data:")
print(nn_model.evaluate(X_train, y_train))
print("Evaluate on validation data:")
print(nn_model.evaluate(X_val, y_val))