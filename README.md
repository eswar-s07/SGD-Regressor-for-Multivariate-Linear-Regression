# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and select input features (first three attributes) and output variables (house price and population).
2. Split the data into training and testing sets.
3. Apply Standard Scaling to both input features and output values.
4. Train a Multi-Output Regression model using SGDRegressor on the training data.
5. Predict the outputs, inverse transform the results, and evaluate performance using Mean Squared Error (MSE).

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Eswar S
RegisterNumber: 212225230067
*/

import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("house.csv")
data.columns = data.columns.str.strip()

X = data[['Size', 'Bedrooms']]
y_price = data['Price']
y_occ = data['Occupants']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)

size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))

new_data = scaler.transform([[size, bed]])
pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```

## Output:
<img width="403" height="107" alt="image" src="https://github.com/user-attachments/assets/730bcebe-191e-4972-8f52-c472065b4a98" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
