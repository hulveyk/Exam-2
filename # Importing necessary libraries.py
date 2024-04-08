# Importing necessary libraries 

import pandas as pd 

import numpy as np 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error, r2_score 

 

# Load the dataset from Excel file 

# Define the new file path 

file_path = "C:/Users/Hulveyk03/Downloads/Restaurant Revenue.xlsx" 

 

# Load the dataset from the new file path 

restaurant_data = pd.read_excel(file_path) 

 

# Display the first few rows of the dataset 

print(restaurant_data.head()) 

 

# Separate features (X) and target variable (y) 

X = restaurant_data[['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 

                     'Average_Customer_Spending', 'Promotions', 'Reviews']] 

y = restaurant_data['Monthly_Revenue'] 

 

# Split the dataset into training and testing sets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

 

# Create and fit the linear regression model 

model = LinearRegression() 

model.fit(X_train, y_train) 

 

# Predict on the testing set 

y_pred = model.predict(X_test) 

 

# Evaluate the model 

mse = mean_squared_error(y_test, y_pred) 

r2 = r2_score(y_test, y_pred) 

print("Mean Squared Error:", mse) 

print("R-squared:", r2) 

 

# Display the coefficients of the model 

coefficients = pd.DataFrame({'Features': X.columns, 'Coefficients': model.coef_}) 

print("\nCoefficients:\n", coefficients) 

 print ( "Happy")
