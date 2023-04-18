import nltk.util

sample_tokens="['Hello everyone.', 'This is an introducton to tokenization.','Sit back, and relax, enjoy your first lesson of NLP]"


#applying bigrams
print("applying bigrams",list(nltk.bigrams(sample_tokens)))


#apply trigrams
print("apply trigrams",list(nltk.trigrams(sample_tokens)))


#applying n-grams
print("applying n-grams",list(nltk.ngrams(sample_tokens,4)))
