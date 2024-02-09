import numpy as np

from constants import *
from BiLSTM import BiLSTM
from LSTM import LSTM
from CNN import CNN
from phoBERTEmbedding import usingPhoBERT

# Dataset Prepare
title_train_ids, text_train_ids, train_labels, title_val_ids, text_val_ids, val_labels, title_test_ids, text_test_ids, test_labels, vocab_size = usingPhoBERT()

# MODEL IMPLEMENTATION AND TRAINING
def startLearning():
  print('TRAINING USING CNN MODEL.......................')
  cnn = CNN(
    title_train_ids,
    text_train_ids,
    train_labels,
    title_val_ids,
    text_val_ids,
    val_labels,
    vocab_size
  )

  cnn.trainModel()
  cnn.testModel(
    [np.array(title_test_ids), np.array(text_test_ids)], 
    np.array(test_labels)
  )
  
  # print('TRAINING USING BiLSTM MODEL......................')
  # bilstm = BiLSTM(
  #   title_train_ids,
  #   text_train_ids,
  #   train_labels,
  #   title_val_ids,
  #   text_val_ids,
  #   val_labels,
  #   vocab_size
  # )

  # bilstm.trainModel()
  # bilstm.testModel(
  #   [np.array(title_test_ids), np.array(text_test_ids)], 
  #   np.array(test_labels)
  # )

  # print('TRAINING USING LSTM MODEL.........................')
  # lstm = LSTM(
  #   title_train_ids,
  #   text_train_ids,
  #   train_labels,
  #   title_val_ids,
  #   text_val_ids,
  #   val_labels,
  #   vocab_size
  # )

  # lstm.trainModel()
  # lstm.testModel(
  #   [np.array(title_test_ids), np.array(text_test_ids)], 
  #   np.array(test_labels)
  # )
  print('TRAINING DONE.............................')