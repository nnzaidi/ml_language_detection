import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
# print(data.head())
# print(data.isnull().sum())
# print(data['language'].value_counts())

# Language Detection Model
x = np.array(data['Text'])
y = np.array(data['language'])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Multiclass classification - will be using the Multinomial Naive Bayes algorithm to solve the problem
model = MultinomialNB()
model.fit(X_train, y_train)
model.score(X_test, y_test) # Accuracy score obtained ~0.95 or 95%

# Try the model to detect language of a text by taking user input
user = input("Enter text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
