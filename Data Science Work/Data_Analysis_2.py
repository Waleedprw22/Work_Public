import numpy as np
import pandas as pd

data = pd.read_csv("https://raw.githubusercontent.com/changyaochen/MECE4520/master/lectures/week-1/iris.csv")
data.head()

Data2 = data[(data.Species=='Iris-versicolor') & (data.PetalLengthCm>0)]

print("The minimum Petal Length (cm) for Iris-color is:" + str(Data2['PetalLengthCm'].min()))
print("The maximum Petal Length (cm) for Iris-color is:" + str(Data2['PetalLengthCm'].max()))


#Part 2

Data_Versi = data[(data.Species=='Iris-versicolor') & (data.SepalWidthCm>0)]
Data_setosa = data[(data.Species=='Iris-setosa') & (data.SepalWidthCm>0)]
Data_Virgin = data[(data.Species=='Iris-virginica') & (data.SepalWidthCm>0)]


print("Iris-setosa has the largest Sepal width of:" + str(Data_setosa['SepalWidthCm'].mean()))
print("Iris-virginia has the second largest Sepal width of:" + str(Data_Virgin['SepalWidthCm'].mean()))
print("Iris-versi  has the smallest Sepal width of:" + str(Data_Versi['SepalWidthCm'].mean()))