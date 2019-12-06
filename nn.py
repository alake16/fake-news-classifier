from preprocessing import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X_train, X_test, y_train, y_test, X_train_fit, vectorizer = preprocessing(False)
 
# Grid Search
def create_model(optimizer="Adamax", init_mode='uniform', activation="relu", neurons=1):
	# define the keras model
	model = Sequential()
	model.add(Dense(units=neurons, activation=activation, kernel_initializer=init_mode, input_dim=len(vectorizer.get_feature_names())))
	model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init_mode))
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
# define the grid search parameters
# batch_size = [128, 256, 512]
# epochs = [2, 4, 6]
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# neurons = [1, 100, 200, 400, 500, 750, 1000, 2000, 4000]
# model = KerasClassifier(build_fn=create_model, epochs=num_epochs, batch_size=num_batches, verbose=1)
# param_grid = dict(neurons=neurons)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
# grid_search = grid.fit(X_train_fit, y_train)
# best_score = grid_search.best_score_
# best_parameters = grid_search.best_params_ 
# print ('Best accuracy: {}, using {}'.format(best_score, best_parameters))

# best parameters from grid search
epochs = 4
batch_size = 256
optimizer = "Adamax"
init_mode = "uniform"
activation = "relu"
neurons = 20000

model = Sequential()
model.add(Dense(units=neurons, activation=activation, kernel_initializer=init_mode, input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init_mode))
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train_fit, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.33)
scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print("Accuracy:", scores[1])
