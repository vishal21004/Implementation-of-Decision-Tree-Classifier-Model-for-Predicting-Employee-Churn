# EX 08 : Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
### Date : 07/10/24
## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VISHAL M.A
RegisterNumber:  212222230177
*/
```
```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv("Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
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
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()

```

## Output:

## HEAD() AND INFO():
![fnl1](https://github.com/vishal21004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119560110/397384ed-8d51-43bb-9105-127fbbbb6c12)


## NULL & COUNT:
![snl2](https://github.com/vishal21004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119560110/908ab6f6-6eed-43c2-8a70-6634597ea5b6)


![fnl3](https://github.com/vishal21004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119560110/df25f53f-a0c6-4955-8c63-723017093fec)


## ACCURACY SCORE:
![fnl4](https://github.com/vishal21004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119560110/8fb3d0e3-c79d-42eb-af24-124101bf0fb9)


## DECISION TREE CLASSIFIER MODEL:
![fnl5](https://github.com/vishal21004/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119560110/b3858bb0-8d65-4359-b409-da27d85590d6)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
