# -*- coding: utf-8 -*-
"""
Aim - Predicting insurance charges for customers.
Problem type - Regression
Source - Kaggle (Medical Cost Personal Datasets)
Algorithms/Techniques applied - Linear Regression, Random Forest Regressor, 
                                 AdaBoost Regressor, Ordinary Least Square (Statsmodel)
Data input for models - Cross-validation and Train-Test splitting 
"""
# Import packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import statsmodels.api as sm
import seaborn as sns

# Read the data
data = pd.read_csv("./Insurance.csv")

# View type of data and its statistical information.
print(data.head(), data.tail(), data.describe())

# View all column names and statistical description of data
print(data.columns)
print(data.info())

# Check for missing values
print(data.isnull().sum()) 

# Converting categorical features' string values to int
data.smoker = [1 if x == 'yes' else 0 for x in data.smoker]
data.sex = [1 if x == 'male' else 0 for x in data.sex]
data.region = pd.get_dummies(data.region)
data.charges = pd.to_numeric(data.charges)
print(data.columns.values)

#-------------------------------------- DATA VISUALIZATION ------------------------------------------------------
# Visualize distribution of values for target variable - 'charges'
plt.figure(figsize=(6,6))
plt.hist(data.charges, bins='auto')
plt.xlabel("charges ->")
plt.title("Distribution of charges values :")

# Generate Box-plots to check for outliers and relation of each feature with 'charges'
cols = ['age', 'children', 'sex', 'smoker', 'region']
for col in cols:
    plt.figure(figsize=(6,6))
    sns.boxplot(x = data[col], y = data['charges'])

# Create Correlation matrix for all features of data.
data.corr()

# Generate heatmap to visualize strong & weak correlations.
sns.heatmap(data.corr(), square = True)

# Generate predictions using all features by a Linear Regression model.
sns.pairplot(data)

#---------------------------- Prepare data for predictive regression models --------------------------------------
y = data.charges.values
X = data.drop(['charges'], axis = 1)   # Drop the target

# --------------------------- PREDICTIVE MODELLING (Call the models to be used) -----------------------------------
rf_reg = RandomForestRegressor(max_features = 'auto', bootstrap = True, random_state = None)
lin_reg = LinearRegression(normalize = True)
ada_reg = AdaBoostRegressor()

# --------------------------- 1st Approach - Using Cross-validation (to avoid overfitting) ------------------------
# --------------------------- Plotting Cross-validation Predictions for each model --------------------------------
# Predict using Random Forest Regressor.
predRF = cross_val_predict(rf_reg, X, y, cv=10)
fig, ax = plt.subplots()
ax.scatter(y, predRF)
ax.plot([y.min(), y.max()], [y.min(), y.max()])
ax.set_xlabel('Actual value ->')
ax.set_ylabel('Predicted value ->')

# Predict using Linear Regression
predLR = cross_val_predict(lin_reg, X, y, cv=10)
fig, ax = plt.subplots()
ax.scatter(y, predLR)
ax.plot([y.min(), y.max()], [y.min(), y.max()])
ax.set_xlabel('Actual value ->')
ax.set_ylabel('Predicted value ->')

# Predict using ADABoost Regressor
predADA = cross_val_predict(ada_reg, X, y, cv=10)
fig, ax = plt.subplots()
ax.scatter(y, predADA)
ax.plot([y.min(), y.max()], [y.min(), y.max()])
ax.set_xlabel('Actual value ->')
ax.set_ylabel('Predicted value ->')

# ------------------ 2nd Approach - Using Train-test-split ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = None)

# Predict using Random Forest Regressor.
rf_reg.fit(X_train, y_train)
predtrainRF = rf_reg.predict(X_train)     # Prediction for train data
predtestRF = rf_reg.predict(X_test)       # Prediction for test data

# Compute R-squared score for both train and test data.
print("R2-score on train data:", r2_score(y_train,predtrainRF))
print("R2-score on test data:", r2_score(y_test, predtestRF))

# Predict using Linear Regression
lin_reg.fit(X_train, y_train)
predtrainL = lin_reg.predict(X_train)
predtestL = lin_reg.predict(X_test)
print("R2-score on train data:",r2_score(y_train, predtrainL))
print("R2-score on test data:",r2_score(y_test, predtestL))

# Predict using XGBoost Regressor
ada_reg.fit(X_train, y_train)
predtrainAda = ada_reg.predict(X_train)
predtestAda = ada_reg.predict(X_test)
print("R2-score on train data:",r2_score(y_train, predtrainAda))
print("R2-score on test data:",r2_score(y_test, predtestAda))

# ------------------ Using Ordinary Least Square from Statsmodel --------------------------------
# ----------------- Allows to view full summary statistics along with p-value and F-statistics ----------------
# On Train data.
X_newtrain = sm.add_constant(X_train)
ols_train = sm.OLS(y_train, X_newtrain)
ols_train_new = ols_train.fit()
print(ols_train_new.summary())

# On Test data.
X_newtest = sm.add_constant(X_test)
ols_test = sm.OLS(y_test, X_newtest)
ols_test_new = ols_test.fit()
print(ols_test_new.summary())   # Produce full statistical summary 

plt.show()
