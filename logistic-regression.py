from preprocessing import preprocessing

X_train, X_test, y_train, y_test, X_train_fit, vectorizer = preprocessing()

y_train = y_train.astype('int')

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C = .5, penalty = 'l1')
model.fit(X_train_fit, y_train)

y_pred = model.predict(vectorizer.transform(X_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.astype('int'), y_pred)

accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[0][1])
print (accuracy)

# uncomment to perform grid search
"""
from sklearn.model_selection import GridSearchCV
parameters = [{
        'C': [.001, .01, .1, .5],
        'penalty': ['l1', 'l2']
        }]
grid_search = GridSearchCV(estimator = model,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10,
                          n_jobs = -1)
grid_search = grid_search.fit(X_train_fit, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_ 

print ('Best Accuacy: ', best_accuracy)
print ('Best Parameters: ', best_parameters)
