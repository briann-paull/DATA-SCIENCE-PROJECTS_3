import pandas as pd
df=pd.read_csv("spam.csv")


print(df.groupby("Category").describe())
df["Spam"]=df["Category"].apply(lambda x:1 if x=="spam" else 0)


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df.Message,df.Spam,test_size=0.2,random_state=1)


from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
X_train_count=v.fit_transform(X_train.values)
X_train_count.toarray()


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

emails = [
    'Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days.',
    'Hey mohan, can we get together to watch footbal game tomorrow?'
]
emails_count = v.transform(emails)
print(model.predict(emails_count))

X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)

 

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])   #it create a flow of precedure to follow