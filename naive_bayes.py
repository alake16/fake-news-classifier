from sklearn.naive_bayes import MultinomialNB
from preprocessing import preprocessing
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test, X_train_fit, vectorizer = preprocessing(True)
clf = MultinomialNB()
alpha = [0.0, 0.25, 0.5, 0.75, 1.0, 5.0, 10.0]
param_grid = dict(alpha=alpha)
scores = ['accuracy', 'precision']

for score in scores:
	grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=score, cv=10,
							   n_jobs=-1)
	grid_search = grid_search.fit(X_train_fit, y_train)
	best_score = grid_search.best_score_
	best_parameters = grid_search.best_params_ 
	print ('Best {}: '.format(score), best_score)
	print ('Best Parameters: ', best_parameters)

clf = MultinomialNB(alpha=0.0)
clf.fit(X_train_fit, y_train)
score = clf.score(vectorizer.transform(X_test), y_test)
print(score)
