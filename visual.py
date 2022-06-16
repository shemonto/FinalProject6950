%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

def unit_plot():
    some_digit_data = X_train[5]
    some_digit_image = some_digit_data.reshape(28, 28)
    
    # imshow plots the matrix
    plt.imshow(some_digit_image, cmap=plt.cm.Blues)
    plt.axis("off")
    plt.show()