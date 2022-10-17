import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import math
import dateutil.parser as parser


numbers = [6, 7, 8, 9, 10, 11, 12]
windpower = [0,0,0,1,80,170,270,410,
640,930,1210,1400,1490,1500,1500,1500,1500,
1500,1500,1500,1500,1500, 1500,1500,1500,1500]
numbers = [6, 7, 8, 9, 10, 11, 12]
numbers = np.array(numbers)
a = np.arange(.5,26,1)
a = np.array(a)
print(len(a))
#a = np.arange(0,26,.5)
#a = np.array(a)
print(len(a))

prob1 = [0] * len(a)
pbar = [0] * len(a)
AEP_1 = [0] * len(a)
cf = [0] * (len(a))

for i in range(0,len(numbers)):
    print("This is for an average wind speed of" + str(numbers[i]))
    c = 1.128 * numbers[i]
#c = 6.768
    for j in range(0, len(a)):
        V = a[j]
        prob1[j] = ((2 * V)/(c**2)) * np.exp(-(V/c) ** 2)

    print(prob1)
    
 

    prob1 = np.array(prob1)
    print("TEST")
    print(len(windpower))
    print(len(prob1))
    print(windpower * prob1)
    #pbar[i] = sum(windpower * prob1) #units of kW
    pbar[i] = np.dot(windpower, prob1)
    print("pbar")
    print(pbar[i])
    AEP_1[i] = pbar[i] * 8760
    print("AEP for average wind speed  " + str(numbers[i]) + ":" + str(AEP_1 ))
    Rated_power = 1.5 * 8760 * 1000 #To convert from MW to KW

    cf[i] = AEP_1[i]/Rated_power
cf_m = [0] * (len(numbers))
for i in range(0,len(numbers)):
    cf_m[i] = cf[i]
print(len(cf_m))
print(len(numbers))
#Wind speed of 7 m/s
plt.scatter(numbers,cf_m)
print(pbar)
cf_m = np.array(cf_m)
print(cf_m)
z = np.polyfit(numbers.flatten(), cf_m.flatten(), 1)
p = np.poly1d(z)
plt.plot(numbers,p(numbers),"r--")
plt.title("y=%.6fx+%.6f"%(z[0],z[1])) 
plt.xlabel("Average wind speed")
plt.ylabel("Cf")
plt.show()
