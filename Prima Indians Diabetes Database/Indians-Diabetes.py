# -*- coding: utf-8 -*-


"""
Created on Thu Apr 28 21:59:47 2022

@author: Naty
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('diabetes.csv')

#Normalizar y limpiar la data

data.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], 
          axis= 1, inplace = True)

data.Age.replace(np.nan, 33, inplace=True)

ranges = [0, 8, 15, 18, 25, 40, 60, 100]

names = ['1', '2', '3', '4', '5', '6', '7']

data.Age = pd.cut(data.Age, ranges, labels=names)

data.dropna(axis=0,how='any', inplace=True)

#Dividir la data en train y test 
data_train = data[:384]
data_test = data[384:]

x = np.array(data_train.drop(['Outcome'], axis=1))
y = np.array(data_test.Outcome)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], axis=1))
y_test_out = np.array(data_test.Outcome)

## REGRESIÓN LOGÍSTICA

logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entrenamiento el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'Accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'Accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'Accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


## MAQUINA DE SOPORTE VECTORIAL

svc = SVC(gamma='auto')

# Entrenamiento el modelo
svc.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# RANDOM FOREST

random_forest = RandomForestClassifier()

# Entrenar el modelo

random_forest.fit(x_train, y_train)

# Metricas

print('*'*50)
print('Random Forest')

# Accuracy de Test de Entrenamiento

print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test, y_test)}')

# Accuracy de Entrenamiento de Entrenamiento

print(f'accuracy de Entrenamiento de Entrenamiento: {random_forest.score(x_train, y_train)}')

# Accuracy de validacion

print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test_out, y_test_out)}')



## ARBOL DE DECISIÓN

arbol = DecisionTreeClassifier()

# Entrenamiento el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')




