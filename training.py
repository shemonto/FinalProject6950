# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 09:45:35 2022

@author: Shemonto Das
"""
from sklearn.neighbors import KNeighborsClassifier

class Train():
    
    def train_predict(self,train_samples,test_samples):
        print("start hoise")
        knn = KNeighborsClassifier()
        knn.fit(train_samples, test_samples)
        knn.predict([train_samples[5]])
        print("sesh hoise")
        return knn