#Applying stemming to the user data
import nltk
from nltk.stem import PorterStemmer, wordnet, WordNetLemmatizer

pst= PorterStemmer()

#applying algorithm PorterStemmer to user data for stemming
print("----------> applying algorithm PorterStemmer to user data for stemming --> ",pst.stem("Giving"), pst.stem("Buying"), pst.stem("Frying"), pst.stem("Coming"))

#Lemmatization
print("----------> Lemmatization")
lemmatizer=WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('omw-1.4')
word_to_stem=["cats","cacti","geese"]
for i in word_to_stem:
  print(i + ":" + lemmatizer.lemmatize(i))

