from sklearn.model_selection import train_test_split
from dl_workshop.logistic_regression import LogisticRegression
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

# Normalize values
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Split train into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Dataframes to numpy arrays
X_train = X_train.values.T
X_val = X_val.values.T
y_train_shaped = y_train.values.reshape(1, y_train.shape[0])
y_val_shaped = y_val.values.reshape(1, y_val.shape[0])
X_test = X_test.values.T

# Create model and fit
lr_model = LogisticRegression(learning_rate=0.01, verbose=False)
lr_model.fit(X_train, y_train_shaped, epochs=1000, validation_data=(X_val, y_val_shaped))

# Get scores
print("Evaluate on training data:")
print(lr_model.evaluate(X_train, y_train_shaped))
print("Evaluate on validation data:")
print(lr_model.evaluate(X_val, y_val_shaped))

# Save model
lr_model.save('models/titanic_model')

# Load model from file
lr_model = LogisticRegression.load('models/titanic_model')

# Train again
lr_model.fit(X_train, y_train_shaped, epochs=8000, validation_data=(X_val, y_val_shaped))

# Get scores
print("From Scratch Model FINAL")
print("Evaluate on training data:")
print(lr_model.evaluate(X_train, y_train_shaped))
print("Evaluate on validation data:")
print(lr_model.evaluate(X_val, y_val_shaped))

# Overwrite best model
lr_model.save('models/titanic_model')

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

