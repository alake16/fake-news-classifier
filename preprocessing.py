import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def fetch_dataset(doing_grid_search):
    df = pd.read_csv("./data/news_dataset.csv")
    df = df.drop(columns=['Unnamed: 0'])
    # drop rows with missing values
    df = df.dropna()
    df['title'] = df['title'].apply(clean_text)
    df['content'] = df['content'].apply(clean_text)
    df.loc[df.label == "fake", 'label'] = 1
    df.loc[df.label == "real", 'label'] = 0
    df['label'] = df['label'].astype('int')
    return split_dataset(doing_grid_search, df)

def split_dataset(doing_grid_search, df):
    if doing_grid_search:
        X_train, X_test, y_train, y_test = train_test_split(df["content"], df["label"],
                                                            test_size = .15, random_state=0)
        # cut training set by 50%
        X_train, X_cut, y_train, y_cut = train_test_split(X_train, y_train, test_size = .50,
                                                          random_state=0)
        # cut test set by 50%
        X_test, X_cut, y_test, y_cut = train_test_split(X_test, y_test, test_size = .50,
                                                          random_state=0)
        return X_train, X_test, y_train, y_test
    else:
        return train_test_split(df["content"], df["label"], test_size = .15, random_state=0)

def clean_text(text):
    # Strip HTML tags
    text = re.sub('<[^<]+?>', ' ', text)
    # Strip escaped quotes
    text = text.replace('\\"', '')
    # Strip quotes
    text = text.replace('"', '')
    return text

def vectorizer(X_train):
    vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), 
                             lowercase=True, max_features=5000)
    X_train_fit = vectorizer.fit_transform(X_train)
    return X_train_fit, vectorizer

def preprocessing(doing_grid_search):
    X_train, X_test, y_train, y_test = fetch_dataset(doing_grid_search)
    X_train_fit, vectorizer_var = vectorizer(X_train)
    return X_train, X_test, y_train, y_test, X_train_fit, vectorizer_var
