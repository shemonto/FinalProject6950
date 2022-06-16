from load_data import *
from visual_data import *
from training import *
from evaluate_data import *
from augmentation import *

my_data = LoadData("mnist_train.csv" , "mnist_test.csv")

x_train, y_train = my_data.get_train_data()

visualization = Visual()

print("-----")
print(x_train)
print("-----")
print(y_train)
#print(x_train[5])

visualization.unit_plot(x_train[5],y_train[5])
visualization.sea_plot(x_train[5],y_train[5])
tr = Train()
knn = tr.train_predict(x_train, y_train)

ev = Evaluate()
ev.eva(x_train, y_train,knn)

ag = Augment()
ag.augment_koro(x_train,y_train)