from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm

X_train, X_test, y_train, y_test, X_train_fit, vectorizer = preprocessing(doing_grid_search=False)

def create_deep_nn(X_train, y_train, vectorizer):
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
	return model

def evaluate_deep_nn(model, vectorizer, X_test, y_test):
	scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
	f = open("./performance/deep_nn.txt", "w+")
	f.write("===== Deep Neural Network Model =====\n")
	i = 0
	for metric in model.metrics_names:
		print("{}: {}".format(metric, scores[i]))
		f.write("{}: {}\n".format(metric, scores[i]))
		i += 1
	f.write("\n\n")
	f.close()

def deep_nn_experiment(X_train, X_test, y_train, y_test, X_train_fit, vectorizer):
	model = create_deep_nn(X_train, y_train, vectorizer)
	evaluate_deep_nn(model, vectorizer, X_test, y_test)

deep_nn_experiment(X_train, X_test, y_train, y_test, X_train_fit, vectorizer)