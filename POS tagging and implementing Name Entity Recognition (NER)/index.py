#Parts of Speech of the English Language

import nltk 
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

sample_string="This semsester has been an absolute disaster"
sample_tokens=word_tokenize(sample_string)


#Parts of Speech
print("----> Parts of Speech ")
for i in sample_tokens:
  print(nltk.pos_tag([i]))  
  print(nltk.pos_tag([i]))
print(" ---> END of parts of Speech")

#NER(NAME ENTITY RECOGNITION)
string="Whoever you are, wherever you may be, the stars of Teyvat always have a place for you in their skies"

string_token=word_tokenize(string)
string_tag=nltk.pos_tag(string_token)
NET=ne_chunk(string_tag)
print("NET NAME ENTITY RECOGNITION",NET)


