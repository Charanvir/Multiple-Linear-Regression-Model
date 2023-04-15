# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Import the dataset
dataset = pd.read_csv("50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encoding the categorical variables
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# Split the dataset into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the multiple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the test set results
y_pred = regressor.predict(x_test)
# Two decimal points
np.set_printoptions(precision=2)
# Reshaping the vector to be vertical
# Once the individuals are vertical, we can to concatenate them horizontally
# Axis 1 = horizontal, 0 = vertical
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

# Get a single prediction
new_startup_prediction = regressor.predict([[1,0,0, 200000, 250000, 450000]])
print(new_startup_prediction)

# Get the final linear regression equations with the values of the coefficient
coefficients = regressor.coef_
intercept = regressor.intercept_

print(coefficients)
print(intercept)
