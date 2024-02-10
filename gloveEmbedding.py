from glove import Corpus, Glove
import pandas as pd
import numpy as np
from underthesea import word_tokenize
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from constants import *
from Nomarlize import normalizeSentence


def getEmbeddingMatrix(tokenizer, vocab_size):
  glove_model = Glove.load(PATH + 'gloveModel.model')
  emb_dict = dict()

  for word in list(glove_model.dictionary.keys()):
    emb_dict[word] = glove_model.word_vectors[glove_model.dictionary[word]]

  emb_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
  for word, index in tokenizer.word_index.items():
    emb_vector = emb_dict.get(word)
    if emb_vector is not None:
      emb_matrix[index] = emb_vector

  return emb_matrix

def getDataIDS(sentences, tokenizer):
  ids = tokenizer.texts_to_sequences(sentences)
  return pad_sequences(ids, padding='post', maxlen=MAX_LEN)

def tokenizeData(title, text):
  print('LOADING GLOVE MODEL......................................')
  # NORMALIZE DATASET
  title = [normalizeSentence(sentence) for sentence in title]
  text = [normalizeSentence(sentence) for sentence in text]

  # TOKENIZE DATASET
  print('TOKENIZING DATASET.......................................')
  title = [word_tokenize(sentence, format='text') for sentence in title]
  text = [word_tokenize(sentence, format='text') for sentence in text]

  return title, text

def prepareData(title, text, tokenizer):
  # MAPPING TO VOCAB
  print('MAPPING AND PADDING DATASET..............................')
  title_ids = getDataIDS(title, tokenizer)
  text_ids = getDataIDS(text, tokenizer)

  return title_ids, text_ids

def getDataset(file_name):
  # GET FROM CSV (ORIGINAL TEXT)
  print('READING DATASET FROM FILE................................')
  file = pd.read_csv(PATH + file_name)

  title = file['Title'].apply(str)
  text = file['Content'].apply(str)

  # GET LABELS
  label = pd.Series([status for status in file['Rating'].apply(int)])
  label = utils.to_categorical(label - 1, num_classes=3)

  return train_test_split(title, text, label, test_size=0.1)

def usingGlove():
  title_train, title_test, text_train, text_test, train_labels, test_labels = getDataset('data.csv')

  title_train, title_val, text_train, text_val, train_labels, val_labels = train_test_split(title_train, text_train, train_labels, test_size=0.1)

  title_train, text_train = tokenizeData(title_train, text_train)
  title_test, text_test = tokenizeData(title_test, text_test)
  title_val, text_val = tokenizeData(title_val, text_val)

  tokenizer = Tokenizer()
  tokenizer.fit_on_texts([title_train, title_val, text_train, text_val, title_test, text_test])

  title_train_ids, text_train_ids = prepareData(title_train, text_train, tokenizer)
  title_val_ids, text_val_ids = prepareData(title_val, text_val, tokenizer)
  title_test_ids, text_test_ids = prepareData(title_test, text_test, tokenizer)

  return title_train_ids, text_train_ids, train_labels, title_val_ids, text_val_ids, val_labels, title_test_ids, text_test_ids, test_labels, tokenizer

def trainGlove():
  print('READING DATASET FILE ...........................')
  data = pd.read_csv(PATH + 'data.csv')
  data = pd.concat([data['Title'].apply(str), data['Content'].apply(str)])

  # Tokenize data

  print('TOKENIZING DATA ...................')
  data = [word_tokenize(normalizeSentence(sentence)) for sentence in data]

  # Training GLOVE model

  EMBEDDING_DIM = 512
  LEARNING_RATE = 0.01

  corpus = Corpus()
  corpus.fit(data, window=15)

  glove = Glove(no_components=EMBEDDING_DIM, learning_rate=LEARNING_RATE)
  glove.fit(corpus.matrix, epochs=100, no_threads=8, verbose=True)
  glove.add_dictionary(corpus.dictionary)

  glove.save(PATH + 'gloveModel.model')
