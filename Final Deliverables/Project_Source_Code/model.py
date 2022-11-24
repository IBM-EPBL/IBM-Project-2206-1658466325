#!/usr/bin/env python
# coding: utf-8
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import pandas as pd
import numpy as np
dt=pd.read_csv("F:/IBM/Heart_Disease_Prediction.csv")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label = le.fit_transform(dt['Heart_Disease'])
label
dt.drop("Heart_Disease", axis=1, inplace=True)
dt["Heart_Disease"] = label
dt
x = dt.drop("Heart_Disease",axis=1)
y = dt["Heart_Disease"]
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.30,random_state=0)
decTree = DecisionTreeClassifier(max_depth=6, random_state=0) #Decision Tree Classifier
decTree.fit(X_train,Y_train)
pickle.dump(decTree, open('decmodel.pkl', 'wb'))
