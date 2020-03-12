#importing packages
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


#visualizing heatmap for correlation
sns.heatmap(new_data.corr())
plt.show()


#importing data
data=pd.read_csv("titanic.csv")

#setting up dependend and independent future
target=data.loc[:,["Survived"]]
data=data.loc[:,["Sex","Survived","Age","Parch","Fare"]]


#onehot encoding
dummies=pd.get_dummies(data.Sex, drop_first=True)
data=data.drop("Sex",axis="columns")
data=pd.concat([data,dummies],axis="columns")


#looking up for null value
print(data.isnull().any())


#fiillna with mean
data.Age=data.fillna(data.Age.mean())
print(data.isnull().any())


#spliting up variable for train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=2)



#creating modle
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))


#predicting model

y_pred=model.predict(X_test)


#probabiity score

probability_score=model.predict_proba(X_test)


