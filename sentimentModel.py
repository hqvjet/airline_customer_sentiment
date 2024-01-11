from glove import Glove
import pandas as pd
import numpy as np
import re
from underthesea import word_tokenize
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split

from constants import *
from BiLSTM import BiLSTM
from LSTM import LSTM
from CNN import CNN
from Nomarlize import normalizeSentence

# Dataset Prepare
def getData(file_name):
  file = pd.read_csv(PATH + file_name)

  title = pd.Series([normalizeSentence(sent) for sent in file['title'].apply(str)])
  text = pd.Series([normalizeSentence(sent) for sent in file['text'].apply(str)])

  return title, text, utils.to_categorical(file['rating'] - 1, num_classes=5)

x_train_title, x_train_text, y_train = getData('train.csv')
x_test_title, x_test_text, y_test = getData('test.csv')

def tokenize_data(title, text):
  arr_title = [word_tokenize(sentence, format='text') for sentence in title]
  arr_text = [word_tokenize(sentence, format='text') for sentence in text]

  return arr_title, arr_text


x_train_title, x_train_text = tokenize_data(x_train_title, x_train_text)

x_train_title, x_val_title, x_train_text, x_val_text, y_train, y_val = train_test_split(x_train_title, x_train_text, y_train, test_size=0.1) 

# Convert to sequences
tokenizer = Tokenizer()

tokenizer.fit_on_texts([x_train_title, x_train_text])

x_train_title_sequence = tokenizer.texts_to_sequences(x_train_title)
x_train_text_sequence = tokenizer.texts_to_sequences(x_train_text)
x_val_title_sequence = tokenizer.texts_to_sequences(x_val_title)
x_val_text_sequence = tokenizer.texts_to_sequences(x_val_text)
x_test_title_sequence = tokenizer.texts_to_sequences(x_test_title)
x_test_text_sequence = tokenizer.texts_to_sequences(x_test_text)

# Padding sequences to the same dimensions
vocab_size = len(tokenizer.word_index) + 1

x_train_title_pad = pad_sequences(x_train_title_sequence, padding='post', maxlen=MAX_LEN)
x_train_text_pad = pad_sequences(x_train_text_sequence, padding='post', maxlen=MAX_LEN)
x_val_title_pad = pad_sequences(x_val_title_sequence, padding='post', maxlen=MAX_LEN)
x_val_text_pad = pad_sequences(x_val_text_sequence, padding='post', maxlen=MAX_LEN)
x_test_title_pad = pad_sequences(x_test_title_sequence, padding='post', maxlen=MAX_LEN)
x_test_text_pad = pad_sequences(x_test_text_sequence, padding='post', maxlen=MAX_LEN)

# GLOVE EMBEDDING IMPLEMENTATION AND USAGE
glove = Glove.load(PATH + 'gloveModel.model')
emb_dict = dict()
word_vectors = glove.word_vectors

for word in list(glove.dictionary.keys()):
  emb_dict[word] = glove.word_vectors[glove.dictionary[word]]

emb_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, index in tokenizer.word_index.items():
  emb_vector = emb_dict.get(word)
  if emb_vector is not None:
    emb_matrix[index] = emb_vector


# MODEL IMPLEMENTATION AND TRAINING
def startLearning():
  cnn = CNN(
    x_train_title_pad,
    x_train_text_pad,
    y_train,
    x_val_title_pad,
    x_val_text_pad,
    y_val,
    vocab_size,
    emb_matrix
  )

  CNN_history = cnn.trainModel()

  cnn.testModel(
    [np.array(x_test_title_pad), np.array(x_test_text_pad)], 
    np.array(y_test)
  )

  lstm = LSTM(
    x_train_title_pad,
    x_train_text_pad,
    y_train,
    x_val_title_pad,
    x_val_text_pad,
    y_val,
    vocab_size,
    emb_matrix
  )

  lstm_history = lstm.trainModel()

  lstm.testModel(
    [np.array(x_test_title_pad), np.array(x_test_text_pad)], 
    np.array(y_test)
  )