import numpy as np

from constants import *
from BiLSTM import BiLSTM
from LSTM import LSTM
from CNN import CNN
from gloveEmbedding import usingGlove, getEmbeddingMatrix
from CNN_BILSTM import CNN_BILSTM

# Dataset Prepare
title_train_ids, text_train_ids, train_labels, title_val_ids, text_val_ids, val_labels, title_test_ids, text_test_ids, test_labels, tokenizer = usingGlove()
vocab_size = len(tokenizer.word_index) + 1
emb_mat = getEmbeddingMatrix(tokenizer, vocab_size)

# MODEL IMPLEMENTATION AND TRAINING
def startLearning():
  # print('TRAINING USING LSTM MODEL.........................')
  # lstm = LSTM(
  #   title_train_ids,
  #   text_train_ids,
  #   train_labels,
  #   title_val_ids,
  #   text_val_ids,
  #   val_labels,
  #   vocab_size,
  #   emb_mat
  # )

  # lstm.trainModel()
  # lstm.testModel(
  #   [np.array(title_test_ids), np.array(text_test_ids)], 
  #   np.array(test_labels)
  # )
  
  # print('TRAINING USING CNN MODEL.......................')
  # cnn = CNN(
  #   title_train_ids,
  #   text_train_ids,
  #   train_labels,
  #   title_val_ids,
  #   text_val_ids,
  #   val_labels,
  #   vocab_size,
  #   emb_mat
  # )

  # cnn_model = cnn.trainModel()
  # cnn.testModel(
  #   [np.array(title_test_ids), np.array(text_test_ids)], 
  #   np.array(test_labels)
  # )
  
  print('TRAINING USING BiLSTM MODEL......................')
  bilstm = BiLSTM(
    title_train_ids,
    text_train_ids,
    train_labels,
    title_val_ids,
    text_val_ids,
    val_labels,
    vocab_size,
    emb_mat
  )

  bilstm_model = bilstm.trainModel()
  bilstm.testModel(
    [np.array(title_test_ids), np.array(text_test_ids)], 
    np.array(test_labels)
  )
  
  # print('TRAINING USING CNN + BiLSTM MODEL.................')
  # cnn_bilstm = CNN_BILSTM(
  #   title_train_ids,
  #   text_train_ids,
  #   train_labels,
  #   title_val_ids,
  #   text_val_ids,
  #   val_labels,
  #   vocab_size,
  #   emb_mat
  # )

  # cnn_bilstm.trainModel()
  # cnn_bilstm.testModel(
  #   [np.array(title_test_ids), np.array(text_test_ids)], 
  #   np.array(test_labels)
  # )

  print('TRAINING DONE.............................')
