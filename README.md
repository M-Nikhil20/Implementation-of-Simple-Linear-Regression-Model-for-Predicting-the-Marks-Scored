# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M Nikhil
RegisterNumber:  212222230095
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
df.head()

df.tail()

X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred

plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()

```

## Output:
![image](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/0a00fe6a-e3b1-4e25-aeab-70dddce34de1)

![image](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/2f7d9d24-34da-476a-9b14-43580448a072)

![image](https://github.com/M-Nikhil20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/c436ddc7-6d86-451e-aa11-c99c416f7722)

![image](https://github.com/M-Nikhil20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/4a4c8618-bd2d-411f-8668-d5920eb5f883)

![image](https://github.com/M-Nikhil20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/6654f829-88d2-4b7b-8ab9-e54aabb92174)

![image](https://github.com/M-Nikhil20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/eec2e065-a686-452e-9882-db6f541e2a1b)

![image](https://github.com/M-Nikhil20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/c0376423-6150-4ba9-bc7e-24fcc44d4054)

![image](https://github.com/M-Nikhil20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/2e947ffe-6bac-4356-9f94-311a851e40b0)

![image](https://github.com/M-Nikhil20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/7197daf2-a3a7-451a-9c8f-a509aab7bfb4)

![image](https://github.com/M-Nikhil20/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707852/ed8b7ba2-6544-4b32-9dfc-e9bab198cc84)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
