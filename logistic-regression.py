from preprocessing import preprocessing

X_train, X_test, y_train, y_test, X_train_fit, vectorizer = preprocessing()

y_train = y_train.astype('int')

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train_fit, y_train)

y_pred = model.predict(vectorizer.transform(X_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.astype('int'), y_pred)

accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[0][1])
print (accuracy)


