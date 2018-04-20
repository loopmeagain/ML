# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('primes.csv')

X = dataset.iloc[:, :-1].values #esta notacion [:,:-1] es un slicing desde el comienzo, hasta el anteultimo elemento (ignora el ultimo)
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
# 3 lineas para el modelo de regrseion linear para fitting de los training sets
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#predictores de resultados
y_pred= regressor.predict(X_test) 
#visualizacion de los resultados del training set
plt.scatter(X_train, y_train,color= "red")
plt.plot(X_train, regressor.predict(X_train),color = "blue")
plt.title("Salary vs Experience (trainingSet)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#visualizacion de los resultados del test set
plt.scatter(X_test, y_test,color= "red")
plt.plot(X_train, regressor.predict(X_train),color = "blue")
plt.title("Salary vs Experience (testSet)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()