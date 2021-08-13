# Importing the libraries
from sklearn.svm import SVC
import numpy as np
import pandas as pd

# cleaning the texts and stemming the texts
import re
import nltk
import pickle  # For saving data into a file
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# importing the dataset
data = pd.read_csv('imdb_dataset.csv')
data = data.iloc[0:5000, :]

# corpus = []
# for i in range(0, data.shape[0]):
#     review = re.sub('[^a-zA-Z]', ' ', data['review'][i])
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     review = [ps.stem(word) for word in review if not word in set(
#         stopwords.words('english'))]
#     review = ' '.join(review)
#     corpus.append(review)

with open('corpusfile', 'rb') as fp:
    corpus = pickle.load(fp)

nn = len(corpus)

# creating the Bag of words Model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0)

scaler = StandardScaler()

# Feature Scaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def convertReviewToArray(rev):
    review = re.sub('[^a-zA-Z]', ' ', rev)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]
    review = ' '.join(review)
    temp = corpus[0:nn]
    temp.append(review)
    cv = CountVectorizer(max_features=1500)
    corp = cv.fit_transform(temp).toarray()
    corp = scaler.transform(corp)
    size = len(temp)
    review = corp[size-1:size]
    temp = temp[0:size-1]
    return review


# Model-5
# FITTING KERNEL SVM TO OUR TRAINING SET
X_train5 = X_train
y_train5 = y_train
X_test5 = X_test

# Training the Kernel SVM model on the Training set
cksvm = SVC(kernel='rbf', random_state=0)
cksvm.fit(X_train5, y_train5)


def predict_review(rev, model):
    arr = convertReviewToArray(rev)
    pred = model.predict(arr)
    return pred


# Saving the data columns from training
pickle.dump(cksvm, open('ml_model.pkl', 'wb'))
