from neural import *
import pandas as pd
from sklearn.model_selection import train_test_split

print("glug")

data = pd.read_csv("Mine-Data.csv")
x = data[["V","H","S"]].values
y = data["M"].values

#train,test=train_test_split(real_data,test_size=.2)
# placeholder_data=[([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

tupleset=[]
for i in range(len(y)):
    tupleset.append((x[i],y[i]))
    print(x[i])

net = NeuralNet(3, 8, 1)
net.train(tupleset)
print(net.test_with_expected(tupleset))