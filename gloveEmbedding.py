from glove import Corpus, Glove
import pandas as pd
import numpy as np
import re
from underthesea import word_tokenize
from constants import *

# """## **PREPARE DATA**"""

def getData(file_name):
  file = pd.read_csv(PATH + file_name)

  title = pd.Series([re.sub(r'\s+', ' ', sent) for sent in file['title'].apply(str)])
  text = pd.Series([re.sub(r'\s+', ' ', sent) for sent in file['text'].apply(str)])

  return pd.concat([title, text])

data = getData('train.csv')

data.info()

# Tokenize data

data = [word_tokenize(sentence) for sentence in data]

# Training GLOVE model

EMBEDDING_DIM = 512
LEARNING_RATE = 0.01

corpus = Corpus()
corpus.fit(data, window = 20)

glove = Glove(no_components=EMBEDDING_DIM, learning_rate=LEARNING_RATE)
glove.fit(corpus.matrix, epochs=100, no_threads=8, verbose=True)
glove.add_dictionary(corpus.dictionary)

word_vectors = glove.word_vectors
word_dictionary = glove.dictionary

glove.save(PATH + 'gloveModel.model')