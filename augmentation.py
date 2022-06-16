# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 16:38:04 2022

@author: Shemonto Das
"""
from scipy.ndimage import shift
import numpy as np

class Augment():
    
    def augment_koro(self,X_train,y_train):
        print("inaugmentation")
        LEFT = [0, -1]
        RIGHT = [0, 1]
        UP = [-1, 0]
        DOWN = [1, 0]
        X_train_aug = X_train.copy()
        y_train_aug = y_train.copy()
        for index, digit in enumerate(X_train[:100]):
            dig = digit.reshape(28, 28)
            dig_label = y_train[index]
            digit_left = shift(dig, LEFT, cval=0).reshape(784)
            digit_right = shift(dig, RIGHT, cval=0).reshape(784)
            digit_up = shift(dig, UP, cval=0).reshape(784)
            digit_down = shift(dig, DOWN, cval=0).reshape(784)
    
            X_train_aug = np.vstack((X_train_aug, digit_left))
            X_train_aug = np.vstack((X_train_aug, digit_right))
            X_train_aug = np.vstack((X_train_aug, digit_up))
            X_train_aug = np.vstack((X_train_aug, digit_down))
            for _ in range(4):
                y_train_aug = np.hstack((y_train_aug, [dig_label]))
        
        print(X_train_aug.shape)
        print(y_train_aug.shape)