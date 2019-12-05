from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers import Dense

X_train, X_test, y_train, y_test, X_train_fit, vectorizer = preprocessing()
 
model = Sequential()
 
model.add(Dense(units=500, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train_fit, y_train, epochs=2, batch_size=128, verbose=1, validation_split=0.33)
scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print("Accuracy:", scores[1])
