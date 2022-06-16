import numpy as np # linear algebra
import pandas as pd

class LoadData:
    def __init__(self, train_path, test_path):
        self.mnist_train = pd.read_csv(train_path)
        self.mnist_test = pd.read_csv(test_path)

        self.y_train = []
        self.x_train = []

        self.y_test = []
        self.x_test = []

        self.cast_to_numpy()

    
    def cast_to_numpy(self):
        self.y_train  = self.mnist_train["label"].copy().to_numpy()
        self.x_train = self.mnist_train.drop(columns=["label"]).to_numpy()

        # print("The training digits data:\n", self.x_train)
        # print("Digit labels: ", self.y_train)

        # Similarly for the test set
        self.y_test = self.mnist_test["label"].copy().to_numpy()
        self.x_test = self.mnist_test.drop(columns=["label"]).to_numpy()

    def get_train_data(self):
        return self.x_train,  self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test