import numpy as np
from constants import *
# from phoBERTEmbedding import extractFeatures
from usePhoBERT import usePhoBERT
from models.BILSTM import BiLSTM
from models.LSTM import LSTM
from models.CNN import CNN
from models.Ensemble_CNN_BILSTM import Ensemble_CNN_BILSTM
from models.Fusion_CNN_BILSTM import Fusion_CNN_BILSTM
from models.Transformer import Transformer
from models.SGD import SGD
from models.KNN import KNN
from models.LOGISTIC_REGRESSION import LOGISTIC_REGRESSION
from models.DECISION_FOREST import DECISION_FOREST
from models.GRU import GRU
from models.BIGRU import BIGRU
from models.Ensemble_BIGRU_CNN import Ensemble_CNN_BIGRU
from models.Fusion_BIGRU_CNN import Fusion_CNN_BIGRU


# Dataset Prepare
title_train_ids, text_train_ids, train_labels, title_val_ids, text_val_ids, val_labels, title_test_ids, text_test_ids, test_labels = usePhoBERT()
# extractFeatures()


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
    )

    cnn_model = cnn.trainModel()
    cnn.testModel(
    [np.array(title_test_ids), np.array(text_test_ids)], 
    np.array(test_labels)
    )

    print('TRAINING USING BiLSTM MODEL......................')
    bilstm = BiLSTM(
    title_train_ids,
    text_train_ids,
    train_labels,
    title_val_ids,
    text_val_ids,
    val_labels,
    )

    bilstm_model = bilstm.trainModel()
    bilstm.testModel(
    [np.array(title_test_ids), np.array(text_test_ids)], 
    np.array(test_labels)
    )

    print('TRAINING USING ENSEMBLE CNN + BiLSTM MODEL.................')
    ensemble_cnn_bilstm = Ensemble_CNN_BILSTM(
        title_train_ids,
        text_train_ids,
        train_labels,
        title_val_ids,
        text_val_ids,
        val_labels,
    )

    ensemble_cnn_bilstm.trainModel()
    ensemble_cnn_bilstm.testModel(
        [np.array(title_test_ids), np.array(text_test_ids)], 
        np.array(test_labels)
    )

    print('TRAINING USING FUSION CNN + BiLSTM MODEL.................')
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

    # print('TRAINING USING TRANSFORMER MODEL.................')
    # transformer_model = Transformer(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # # transformer_model.trainModel()
    # transformer_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    # print('TRAINING USING SGD MODEL.................')
    # sgd_model = SGD(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # sgd_model.trainModel()
    # sgd_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    # print('TRAINING USING KNN MODEL.................')
    # knn_model = KNN(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # knn_model.trainModel()
    # knn_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    # print('TRAINING USING LOGISTIC REGRESSION MODEL.................')
    # lg_model = LOGISTIC_REGRESSION(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # # lg_model.trainModel()
    # lg_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    # print('TRAINING USING DECISION FOREST MODEL.................')
    # df_model = DECISION_FOREST(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # df_model.trainModel()
    # df_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    # print('TRAINING USING GRU MODEL.................')
    # gru_model = GRU(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # gru_model.trainModel()
    # gru_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    # print('TRAINING USING BIGRU MODEL.................')
    # bigru_model = BIGRU(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # bigru_model.trainModel()
    # bigru_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    # print('TRAINING USING ENSEMBLE CNN + BIGRU MODEL.................')
    # ensemble_cnn_bigru_model = Ensemble_CNN_BIGRU(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # ensemble_cnn_bigru_model.trainModel()
    # ensemble_cnn_bigru_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    # print('TRAINING USING FUSION CNN + BIGRU MODEL.................')
    # fusion_cnn_bigru_model = Fusion_CNN_BIGRU(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    # )
    #
    # fusion_cnn_bigru_model.trainModel()
    # fusion_cnn_bigru_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )
