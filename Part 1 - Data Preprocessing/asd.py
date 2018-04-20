# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("Data.csv")

matrizVariablesIndependientes=dataset.iloc[:,:-1].values # significa que del dataset agarra desde el comienzo hasta el la ante ultima columna
vectorVariablesDependientes=dataset.iloc[:,3].values

#esto es para manejar la falta de datos*
#from sklearn.preprocessing import Imputer
#imputer= Imputer(missing_values="NaN", strategy="mean", axis=0)
#imputer=imputer.fit(matrizVariablesIndependientes[:,1:3])
#matrizVariablesIndependientes[:,1:3]= imputer.transform(matrizVariablesIndependientes[:,1:3])
#print(matrizVariablesIndependientes)

# esto es para manejar datos que tienen que estar encodeados
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#label_encoder_x= LabelEncoder()
#matrizVariablesIndependientes[:,0]=label_encoder_x.fit_transform(matrizVariablesIndependientes[:,0])
#el hot encoder me sirve para transformar en variables dummys (tabla de verdad/mux)
#hotEncoder= OneHotEncoder(categorical_features=[0])
#matrizVariablesIndependientes= hotEncoder.fit_transform(matrizVariablesIndependientes).toarray()


#label_encoder_y= LabelEncoder()
#vectorVariablesDependientes=label_encoder_y.fit_transform(vectorVariablesDependientes)
#splitting de dataset
from sklearn.cross_validation import train_test_split
indep_train,indep_test,dep_train,dep_test= train_test_split(matrizVariablesIndependientes,vectorVariablesDependientes, test_size=0.2,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
indep_train=sc_x.fit_transform(indep_train)
indep_test=sc_x.transform(indep_test)"""
