### Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
# AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

# Equipments Required:
# Hardware – PCs

Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1.Prepare your data Clean and format your data Split your data into training and testing sets

2.Define your model Use a sigmoid function to map inputs to outputs Initialize weights and bias terms

3.Define your cost function Use binary cross-entropy loss function Penalize the model for incorrect predictions

4.Define your learning rate Determines how quickly weights are updated during gradient descent

5.Train your model Adjust weights and bias terms using gradient descent Iterate until convergence or for a fixed number of iterations

6.Evaluate your model Test performance on testing data Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters Experiment with different learning rates and regularization techniques

8.Deploy your model Use trained model to make predictions on new data in a real-world application.

# Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:VAISHNAVI S
Register Number:212222230165  
```
```
import pandas as pd
df=pd.read_csv("/content/Employee.csv")

df.head()

df.info()

df.isnull().sum()

df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["salary"]=le.fit_transform(df["salary"])
df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
# Output:

# Initial data set:
![Screenshot 2023-10-21 214541](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541897/90396fa4-2d45-476e-b4c5-a1bc8ea7dede)


# Data info:

![Screenshot 2023-10-21 214547](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541897/0d96642d-9169-4715-9ad0-289498550848)

# Optimization of null values:


![Screenshot 2023-10-21 214552](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541897/34299776-7f8e-474d-939c-eecaefb9bfac)

# Assignment of x and y values:
![Screenshot 2023-10-21 214558](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541897/bb32dfe6-3fc7-40fa-a54d-31103ee16656)



# Converting string literals to numerical values using label encoder:
![Screenshot 2023-10-21 214605](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541897/c759b021-2695-4d24-942c-ef7a38e7fe7b)


# Accuracy:
![Screenshot 2023-10-21 214611](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541897/dd8883bf-f686-4e55-813a-b13e25af1ded)


# Prediction:

![Screenshot 2023-10-21 220739](https://github.com/Vaishnavi-saravanan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118541897/2a52c48d-3ccf-417b-935b-e1de935e628a)

# Result:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
