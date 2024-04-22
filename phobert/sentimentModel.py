import numpy as np

from constants import *
from BiLSTM import BiLSTM
from LSTM import LSTM
from CNN import CNN
# from phoBERTEmbedding import extractFeatures
from usePhoBERT import usePhoBERT
from Ensemble_CNN_BILSTM import Ensemble_CNN_BILSTM
from Fusion_CNN_BILSTM import Fusion_CNN_BILSTM


# Dataset Prepare
title_train_ids, text_train_ids, train_labels, title_val_ids, text_val_ids, val_labels, title_test_ids, text_test_ids, test_labels = usePhoBERT()
# extractFeatures()


# MODEL IMPLEMENTATION AND TRAINING
def startLearning():
  # print('TRAINING USING CNN MODEL.......................')
  # cnn = CNN(
  #   title_train_ids,
  #   text_train_ids,
  #   train_labels,
  #   title_val_ids,
  #   text_val_ids,
  #   val_labels,
  # )
  #
  # cnn_model = cnn.trainModel()
  # cnn.testModel(
  #   [np.array(title_test_ids), np.array(text_test_ids)], 
  #   np.array(test_labels)
  # )
  
  # print('TRAINING USING BiLSTM MODEL......................')
  # bilstm = BiLSTM(
  #   title_train_ids,
  #   text_train_ids,
  #   train_labels,
  #   title_val_ids,
  #   text_val_ids,
  #   val_labels,
  # )
  #
  # bilstm_model = bilstm.trainModel()
  # bilstm.testModel(
  #   [np.array(title_test_ids), np.array(text_test_ids)], 
  #   np.array(test_labels)
  # )
  
    # print('TRAINING USING ENSEMBLE CNN + BiLSTM MODEL.................')
    # ensemble_cnn_bilstm = Ensemble_CNN_BILSTM(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # ensemble_cnn_bilstm.trainModel()
    # ensemble_cnn_bilstm.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    print('TRAINING USING ENSEMBLE CNN + BiLSTM MODEL.................')
    fusion_cnn_bilstm = Fusion_CNN_BILSTM(
        title_train_ids,
        text_train_ids,
        train_labels,
        title_val_ids,
        text_val_ids,
        val_labels,
    )

    fusion_cnn_bilstm.trainModel()
    fusion_cnn_bilstm.testModel(
        [np.array(title_test_ids), np.array(text_test_ids)], 
        np.array(test_labels)
    )

  # print('TRAINING USING LSTM MODEL.........................')
  # lstm = LSTM(
  #   title_train_ids,
  #   text_train_ids,
  #   train_labels,
  #   title_val_ids,
  #   text_val_ids,
  #   val_labels,
  # )
  #
  # lstm.trainModel()
  # lstm.testModel(
  #   [np.array(title_test_ids), np.array(text_test_ids)], 
  #   np.array(test_labels)
  # )
  # print('TRAINING DONE.............................')
