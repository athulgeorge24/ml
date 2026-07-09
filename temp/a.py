import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

marks=np.array([9,11,13,15,17])

x=[1,2,3,4]
y=[6,10,19,24]

# numpy

print("numpy")  
print("\n")  

marks1=marks.reshape(-1,1)
print("Marks: ", marks)
print("Avg Marks:", np.mean(marks))
print("reshaped :\n",marks1)
rand_num= np.random.randint(1,100,size=5) 
print("Random Data:", rand_num)
zeros=np.zeros(6)
ones=np.ones(6)
rand=np.random.random(6)

print(zeros)
print(ones)
print(rand)

zeros1=zeros.reshape(2,3)
ones1=ones.reshape(2,3)
rand1=rand.reshape(2,3)

print(zeros1)
print(ones1)
print(rand1)

print("\n")  

print("\n")  

print("matplotlab")  

#matplotlab
plt.plot(x,y,color='red', linestyle='-.',label='Line Trend')
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.show()
plt.scatter(x, y, color='blue', marker='x',label='Actual Data')
plt.show()

print("\n")  

print("pandas")  

#pandas
df=pd.read_csv("data.csv")
print(df.head())     
print("\n")  

df["Hours_studies"] = df["Hours_studies"].replace(6, 5)

print("\n")
print(df.head())     










'''

#linear regresion
x1=df[["Hours_studies"]]
y1=df["score"]

model = LinearRegression() #model making 

model.fit(x1, y1)

print(f"Learned Weight (w): {model.coef_[0]:.2f}")
print(f"Learned Bias   (b): {model.intercept_:.2f}")
'''