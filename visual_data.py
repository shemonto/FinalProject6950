import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class Visual:
    
    def unit_plot(self,train_sample,test_sample):
        some_digit_data = train_sample
        some_digit_image = some_digit_data.reshape(28, 28)
        
        # imshow plots the matrix
        plt.imshow(some_digit_image, cmap=plt.cm.Blues)
        #plt.axis("off")
        plt.xlabel("X  pixel label")
        plt.ylabel("pixel label")
        plt.title('First Two Dimensions of Projected Data ')
        plt.show()
        print(test_sample)
        
        
    def sea_plot(self,train_sample,test_sample):
        sns.set()
        some_digit_data = train_sample
        some_digit_image = some_digit_data.reshape(28, 28)
        plt.plot(some_digit_image)
        plt.title('2D spread of data using seaborn ')
        plt.xlabel("X  pixel label")
        plt.ylabel("pixel label")
        plt.show()