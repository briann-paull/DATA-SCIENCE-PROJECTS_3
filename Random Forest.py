#importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
digits=load_digits()
print(dir(digits))

plt.gray()
plt.matshow(digits.images[1])
plt.show()


df = pd.DataFrame(digits.data)
df.head()

df['target'] = digits.target

X = df.drop('target',axis='columns')
y = df.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)#nestimator means number of tree
model.fit(X_test,y_test)


print(model.score(X_test, y_test))

y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)




import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')