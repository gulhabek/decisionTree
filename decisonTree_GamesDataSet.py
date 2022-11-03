# Making imports
import pandas as pd
import numpy as np
from sklearn import tree

# load data set
data=pd.read_csv("games.csv")

# Separating the Data Set into Dependent and Independent Attributes
#X=data.iloc[:,[1,4,5]]
X=data.filter(items=['rated', 'turns', 'victory_status'])
y=data.filter(items=['winner'])

#one-hot encoder
X_oh=pd.get_dummies(X, columns=['rated', 'victory_status'])

# Separating Data as Training and Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_oh, y, test_size=0.20, random_state=0)

# Creating and Training a Decision Tree Model
classifier=tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
classifier.fit(X_train, y_train)

# Making prediction
y_pred=classifier.predict(X_test)

import sklearn.metrics as mt
mt.accuracy_score(y_test, y_pred)

#visualize
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10), dpi=100)
tree.plot_tree(classifier, rounded=True, filled=True, impurity=True)