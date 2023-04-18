#finding the frequency of the number of times a word has been used in the sample data
from nltk.probability import FreqDist
fdist=FreqDist()

#finding the top 5 most common tokens in the sample data
top_5=fdist.most_common(10)
top_5
