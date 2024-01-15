import numpy as np

from constants import *
from BiLSTM import BiLSTM
from LSTM import LSTM
from CNN import CNN
from phoBERTEmbedding import usingPhoBERT

# Dataset Prepare
title_train_ids, text_train_ids, train_labels, title_val_ids, text_val_ids, val_labels, title_test_ids, text_test_ids, test_labels = usingPhoBERT()

# MODEL IMPLEMENTATION AND TRAINING
def startLearning():
  cnn = CNN(
    title_train_ids,
    text_train_ids,
    train_labels,
    title_val_ids,
    text_val_ids,
    val_labels,
  )

  CNN_history = cnn.trainModel()

  cnn.testModel(
    [np.array(title_test_ids), np.array(text_test_ids)], 
    np.array(test_labels)
  )