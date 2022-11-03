# Making imports
import numpy as np
import pandas as pd
import matplotlib as plt

# load data set
data=pd.read_csv("SosyalMedyaReklamKampanyasi.csv")

# Separating the Data Set into Dependent and Independent Attributes
X=data.iloc[:,[2,3]]
y=data.iloc[:,4]

# Separating Data as Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.tree import DecisionTreeClassifier 
classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth=(5), random_state=0)
classifier.fit(X_train, y_train)

# Making prediction
y_pred=classifier.predict(X_test)

from sklearn import tree
tree.plot_tree(classifier)
# confusion_matrix
import sklearn.metrics as mt
mt.accuracy_score(y_test,y_pred)
