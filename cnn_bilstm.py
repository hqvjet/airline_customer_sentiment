from glove import Corpus, Glove
import pandas as pd
import numpy as np
import re
from underthesea import word_tokenize
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Flatten, Dense
from tensorflow.keras import utils
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, concatenate
from keras.models import Model

path = 'resources/'


# Dataset Prepare
def getData(file_name):
  file = pd.read_csv(path + file_name)

  title = pd.Series([re.sub(r'\s+', ' ', sent) for sent in file['title'].apply(str)])
  text = pd.Series([re.sub(r'\s+', ' ', sent) for sent in file['text'].apply(str)])

  return title, text, utils.to_categorical(file['rating'] - 1, num_classes=5)


x_train_title, x_train_text, y_train = getData('train.csv')
x_test_title, x_test_text, y_test = getData('test.csv')


def tokenize_data(title, text):
  arr_title = [word_tokenize(sentence, format='text') for sentence in title]
  arr_text = [word_tokenize(sentence, format='text') for sentence in text]

  return arr_title, arr_text


x_train_title, x_train_text = tokenize_data(x_train_title, x_train_text)

tokenizer = Tokenizer()

tokenizer.fit_on_texts([x_train_title, x_train_text])

x_train_title_sequence = tokenizer.texts_to_sequences(x_train_title)
x_train_text_sequence = tokenizer.texts_to_sequences(x_train_text)
x_test_title_sequence = tokenizer.texts_to_sequences(x_test_title)
x_test_text_sequence = tokenizer.texts_to_sequences(x_test_text)

vocab_size = len(tokenizer.word_index) + 1
MAX_LEN = 512

x_train_title_pad = pad_sequences(x_train_title_sequence, padding='post', maxlen=MAX_LEN)
x_train_text_pad = pad_sequences(x_train_text_sequence, padding='post', maxlen=MAX_LEN)
x_test_title_pad = pad_sequences(x_test_title_sequence, padding='post', maxlen=MAX_LEN)
x_test_text_pad = pad_sequences(x_test_text_sequence, padding='post', maxlen=MAX_LEN)

glove = Glove.load(path + 'gloveModel.model')
emb_dict = dict()

for word in list(glove.dictionary.keys()):
  emb_dict[word] = glove.word_vectors[glove.dictionary[word]]

emb_matrix = np.zeros((vocab_size, MAX_LEN))
for word, index in tokenizer.word_index.items():
  emb_vector = emb_dict.get(word)
  if emb_vector is not None:
    emb_matrix[index] = emb_vector

EPOCH = 100
BATCH_SIZE = 4

# """## **LSTM**
# embedding_dim = 128
# hidden_size = 256

# # Đầu vào cho title
# title_input = Input(shape=(x_train_title_pad.shape[1],))
# title_embedding = Embedding(vocab_size, embedding_dim, input_length=x_train_title_pad.shape[1])(title_input)
# title_lstm = LSTM(hidden_size, return_sequences=True)(title_embedding)
# title_lstm_dropout = Dropout(0.2)(title_lstm)
# title_lstm_final = LSTM(hidden_size)(title_lstm_dropout)

# # Đầu vào cho text
# text_input = Input(shape=(x_train_text_pad.shape[1],))
# text_embedding = Embedding(vocab_size, embedding_dim, input_length=x_train_text_pad.shape[1])(text_input)
# text_lstm = LSTM(hidden_size, return_sequences=True)(text_embedding)
# text_lstm_dropout = Dropout(0.2)(text_lstm)
# text_lstm_final = LSTM(hidden_size)(text_lstm_dropout)

# # Kết hợp hai đầu vào
# combined = concatenate([title_lstm_final, text_lstm_final])

# # Các bước còn lại của mô hình
# dense1 = Dense(128, activation='relu')(combined)
# output = Dense(y_train.shape[1], activation='softmax')(dense1)

# # Xây dựng mô hình
# model_LSTM = Model(inputs=[title_input, text_input], outputs=output)

# model_LSTM.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_LSTM.summary()

# from keras.utils import plot_model

# plot_model(model_LSTM, to_file='model.png', show_shapes=True, show_layer_names=True)

# model_LSTM_history = model_LSTM.fit(
#     [np.array(x_train_title_pad), np.array(x_train_text_pad)],
#     y_train,
#     epochs=EPOCH,
#     batch_size=4,
#     verbose=1,
#     validation_data=([np.array(x_test_title_pad), np.array(x_test_text_pad)], y_test)
# )

# """# **CNN**"""


# """# **BiLSTM + CNN**"""

from keras.layers import Input, concatenate, Dense, Concatenate, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D
from keras.models import Model
from keras import backend as K


def build_ensemble_model(model_BiLSTM, model_CNN):
    # Define input layers for the title and text inputs
    title_input = Input(shape=(x_train_title_pad.shape[1],))
    text_input = Input(shape=(x_train_text_pad.shape[1],))

    # Get the predictions from the BiLSTM model
    lstm_predictions = model_BiLSTM([title_input, text_input])

    # Get the predictions from the CNN model
    cnn_predictions = model_CNN([title_input, text_input])

    # Concatenate the predictions
    concatenated_predictions = Concatenate()([lstm_predictions, cnn_predictions])

    # Add a dense layer
    dense_layer = Dense(64, activation='relu')(concatenated_predictions)

    # Add another dense layer for the final output
    output_layer = Dense(5, activation='softmax')(dense_layer)

    ensemble_model = Model(inputs=[title_input, text_input], outputs=output_layer)

    return ensemble_model


# Build the ensemble model
model_ensemble_bilstm_cnn = build_ensemble_model(model_BiLSTM, model_CNN)
model_ensemble_bilstm_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ensemble_bilstm_cnn.summary()

history = model_ensemble_bilstm_cnn.fit(
    [np.array(x_train_title_pad), np.array(x_train_text_pad)],
    y_train,
    epochs=EPOCH,
    batch_size=4,
    verbose=1,
    validation_data=([np.array(x_test_title_pad), np.array(x_test_text_pad)], y_test)
)