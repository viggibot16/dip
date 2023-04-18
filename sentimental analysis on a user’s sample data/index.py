import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

dataset = {
    "I liked the movie" : "positive",
    "Heroe's acting is bad but heroine looks good" : "negative",
    "It's a good movie. Nice story" : "positive",
    "Overall nice movie" : "positive",
    "Nice songs. But sadly boring ending." : "negative",
    "Sad movie, boring movie" : "negative"
    
}

# creating bag of words model

dataset = pd.DataFrame(list(dataset.items()))
print(dataset)
dataset.columns = ["Text", "Reviews"]
nltk.download('stopwords')


corpus = []

 

for i in range(0, 6):

    text = re.sub('[^a-zA-Z]', '', dataset['Text'][i])
    
    text = text.lower()

    text = text.split()

    ps = PorterStemmer()

    text = ''.join(text)

    corpus.append(text)

cv = CountVectorizer(max_features = 1500)
print("creating bag of words model",cv)

# splitting the data set into training set and test set

print("---> # splitting the data set into training set and test set")
X = cv.fit_transform(corpus).toarray()

print("X",X)

y = dataset.iloc[:, 1].values
print("y",y)






