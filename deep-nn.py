from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm

X_train, X_test, y_train, y_test, X_train_fit, vectorizer = preprocessing(False)
 
model = Sequential()

inputDim = len(vectorizer.get_feature_names())
 
model.add(Dropout(0.2, input_shape=(inputDim,)))
model.add(Dense(units=200, activation='relu', input_dim=inputDim, kernel_constraint=maxnorm(3)))
model.add(Dense(units=100, activation='relu', input_dim=inputDim, kernel_constraint=maxnorm(3)))
model.add(Dense(units=50, activation='relu', input_dim=inputDim, kernel_constraint=maxnorm(3)))
model.add(Dense(units=100, activation='relu', input_dim=inputDim, kernel_constraint=maxnorm(3)))
model.add(Dense(units=50, activation='relu', input_dim=inputDim, kernel_constraint=maxnorm(3)))
model.add(Dense(units=1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train_fit, y_train, epochs=5, batch_size=128, verbose=1, validation_split=0.33)
scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print("Accuracy:", scores[1])


