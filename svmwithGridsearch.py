# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 19:26:13 2025

@author: asus
"""


from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import classification_report
digit=load_digits()
X=digit.data
y=digit.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
svc=SVC()
svm_params={"C":[0.1,1,10],"kernel":["rbf","lineer"],"gamma":["scale","auto"]}

grid_search_svm=GridSearchCV(svc, svm_params,cv=5)
grid_search_svm.fit(X_train,y_train)
print("en iyi parametreler:",grid_search_svm.best_params_)
print("en iyi score:",grid_search_svm.best_score_)

test_score = grid_search_svm.score(X_test, y_test)
print("Test doğruluğu:", test_score)
