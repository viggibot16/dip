import nltk
import nltk.corpus
from nltk.tokenize import word_tokenize

#taking sample data from user
nltk.download('punkt')
sample="['Hello everyone.', 'This is an introducton to tokenization.','Sit back, and relax, enjoy your first lesson of NLP]"

#Applying tokenization to the user example
sample_tokens=word_tokenize(sample)
print(sample_tokens)

#finding the type and the length of the string
print(type(sample_tokens), len(sample_tokens))

