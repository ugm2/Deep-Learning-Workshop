import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from LogisticRegression.logistic_regression import LogisticRegression
import pandas as pd


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
y_train = y_train.values
y_val = y_val.values
X_test = X_test.values.T

# Create model and fit
lr_model = LogisticRegression(learning_rate=0.01, num_iter=1000, verbose=True)
lr_model.fit(X_train, y_train, validation_data=(X_val, y_val))

# Get scores
print('Train accuracy:', lr_model.score(X_train, y_train))
print('Validation accuracy:', lr_model.score(X_val, y_val))

# Save model
lr_model.save('models/titanic_model.npy')

# Load model from file
lr_model = LogisticRegression(learning_rate=0.01, num_iter=8000, verbose=True)
lr_model.load('models/titanic_model.npy')

# Train again
lr_model.fit(X_train, y_train, validation_data=(X_val, y_val))

# Get scores
print('Train accuracy:', lr_model.score(X_train, y_train))
print('Validation accuracy:', lr_model.score(X_val, y_val))

# Overwrite best model
lr_model.save('models/titanic_model.npy')

### THIS CODE IS FOR SUBMITTING RESULTS TO TITANIC KAGGLE COMPETITION ###
# # Predict test set
# predictions = lr_model.predict(X_test)[0]

# # Convert True/False to 1/0
# predictions = np.array([1 if x == True else 0 for x in predictions])

# # Save CSV
# submission_df = pd.DataFrame({
#     'PassengerId': test.index,
#     'Survived': predictions
# })
# submission_df.to_csv('data/submission.csv', index=False)
