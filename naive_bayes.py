from sklearn.naive_bayes import MultinomialNB
from preprocessing import preprocessing

X_train, X_test, y_train, y_test, X_train_fit, vectorizer = preprocessing()
print(y_train)
clf = MultinomialNB()
clf.fit(X_train_fit, y_train)