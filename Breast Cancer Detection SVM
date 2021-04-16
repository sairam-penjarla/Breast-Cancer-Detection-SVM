import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
sns.set()
data = load_breast_cancer()
x = pd.DataFrame(data['data'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
#print("shapes of x_train,x_test,y_train,y_test are",X_train.shape, X_test.shape, y_train.shape, y_test.shape)
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred)*100,'% accuracy')
print('---------After Hyperparameter Tunning----------')
x_train_min = X_train.min()
x_train_range = (X_train - x_train_min).max()
x_train_final = (X_train - x_train_min)/x_train_range
X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_final = (X_test - X_test_min)/X_test_range

clf.fit(x_train_final, y_train)
y_pred = clf.predict(X_test_final)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred)*100,'% accuracy')
ab = sns.heatmap(cm, annot=True, linewidths=.5, cmap="YlGnBu",xticklabels = ['predictedAsHasCancer','predictedAsNoCancer'] ,yticklabels=['hasCancer','noCancer'])
plt.show()
