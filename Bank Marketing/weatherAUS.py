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


data = pd.read_csv("weatherAUS.csv")

#Normalizar y limpiar la data

data.RainToday.replace(['Yes', 'No'], [1, 0], inplace=True)
data.RainToday.value_counts()

data.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)
data.RainTomorrow.value_counts()

data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm'], axis=1, inplace=True)

data.dropna(axis=0, how='any', inplace=True)

#Dividir la data en train y test
