# Import necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

# Load the 20 newsgroups dataset
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# Extract features from the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

# Define the classifiers
# logreg = LogisticRegression(random_state=42)
# bnb = BernoulliNB()
mnb = MultinomialNB()
# svm = LinearSVC(random_state=42)
# sgd = SGDClassifier(random_state=42)

# Train and evaluate each classifier
for clf in [mnb]:
    print("--->",clf)
    clf.fit(X_train, data_train.target)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__)
    print(classification_report(data_test.target, y_pred))
    print()
