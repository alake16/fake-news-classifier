from preprocessing import preprocessing

X_train, X_test, y_train, y_test, X_train_fit, vectorizer = preprocessing(False)

y_train = y_train.astype('int')

import xgboost as xgb
model = xgb.XGBClassifier(min_child_weight = 0.1, gamma = 5, max_depth = 4) # min_child_weight = 0.1, gamma = 5, max_depth = 4
model.fit(X_train_fit, y_train)

y_pred = model.predict(vectorizer.transform(X_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.astype('int'), y_pred)

accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[0][1])
print("Accuracy: {}".format(accuracy))
f = open("./performance/xgboost.txt", "w+")
f.write("===== XGBoost Model =====\n")
f.write("Accuracy: {}\n".format(accuracy))
f.write("\n\n")
f.close()
# uncomment to perform grid search
"""
from sklearn.model_selection import GridSearchCV
parameters = [{
        'min_child_weight': [.1, .001],
     #   'gamma': [4, 5]
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
"""
