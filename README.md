# fake-news-classifier

## Summary
Exploration of multiple ML algorithms for detecting fake news in the real world.

## Dataset
The two datasets used to create the larger news dataset were both found on Kaggle.com. The kernel where the combining of the two datasets was can be found [Here](https://www.kaggle.com/anthonyc1/fake-news-classifier-final-project "Here")

## Preprocessing
The preprocessing tasks involved with the news dataset include:

* Dropping unused/unnecessary features for our task
	* Abritrary ID
	* Publication address
	* Article title
* Cleaning up article content text
	* Removing any HTML tags
	* Stripping quotes and escaped quotes
* Creating a bag of words and vectorizer
	* Discarding stop words
	* Creating n-grams
	* Converting train and test sets into vectorized form
* Splitting training and testing sets accordingly
	* Accounting for dataset size when performing grid search on hyperparameters

## Models
The following algorithms were used to train a model: 

* Neural Networks
	* Shallow (no hidden layers)
	* Deep (multiple hidden layers)
* Logistic Regression
* Naive Bayes
* XGBoost

Each model implementation is trained on uniform training and testing datasets. Each model is also tuned for the given hyperparameters that apply.

