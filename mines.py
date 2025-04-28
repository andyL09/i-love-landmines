from neural import *
import pandas as pd
from sklearn.model_selection import train_test_split

print("glug")

# data = pd.read_csv("Mine_Dataset.csv")
# x = data[["V","H","S"]].values
# y = data["M"].values

#xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)

placeholder_data=[([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

net = NeuralNet(2, 3, 1)
net.train(placeholder_data)
print(net.test_with_expected(placeholder_data))