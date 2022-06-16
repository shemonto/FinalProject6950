# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:50:11 2022

@author: Shemonto Das
"""


from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Evaluate():
    
    def eva(self,X_train,y_train,knn):
        print("in evaluate")
        knn_preds = cross_val_predict(knn, X_train, y_train, cv=3)
        knn_preds
        self.confusion_matrix(y_train, knn_preds)

    def confusion_matrix(self,y_train, knn_preds):
        cf_mat = confusion_matrix(y_train, knn_preds)
        cm_fig, cm_ax = plt.subplots(figsize=(10, 10))
        cf_mat_disp = ConfusionMatrixDisplay(cf_mat)
        cf_mat_disp.plot(ax=cm_ax, cmap=plt.cm.Blues)
        plt.show()
#conda install scikit-learn=0.17
