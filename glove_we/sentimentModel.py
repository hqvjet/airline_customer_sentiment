import numpy as np

from constants import *
from models.BILSTM import BiLSTM
from models.LSTM import LSTM
from models.CNN import CNN
from gloveEmbedding import usingGlove2, getEmbeddingMatrix, attachEmbeddingToIds
from models.Ensemble_CNN_BILSTM import CNN_BILSTM as En_CNN_BILSTM
from models.Fusion_CNN_BILSTM import CNN_BILSTM as Fu_CNN_BILSTM
from models.Ensemble_BIGRU_CNN import Ensemble_CNN_BIGRU
from models.Fusion_BIGRU_CNN import Fusion_CNN_BIGRU
from models.GRU import GRU
from models.BIGRU import BIGRU
from models.Transformer import Transformer
from models.KNN import KNN
from models.LOGISTIC_REGRESSION import LOGISTIC_REGRESSION
from models.SGD import SGD
from models.DECISION_FOREST import DECISION_FOREST

# Dataset Prepare
title_train_ids, text_train_ids, train_labels, title_val_ids, text_val_ids, val_labels, title_test_ids, text_test_ids, test_labels, tokenizer = usingGlove2()
vocab_size = len(tokenizer.word_index) + 1
emb_mat = getEmbeddingMatrix(tokenizer, vocab_size)

# # MODEL IMPLEMENTATION AND TRAINING
def startLearning():
    # print('TRAINING USING LSTM MODEL.........................')
    # lstm = LSTM(
    # title_train_ids,
    # text_train_ids,
    # train_labels,
    # title_val_ids,
    # text_val_ids,
    # val_labels,
    # vocab_size,
    # emb_mat
    # )
    #
    # lstm.trainModel()
    # lstm.testModel(
    # [np.array(title_test_ids), np.array(text_test_ids)], 
    # np.array(test_labels)
    # )
    #
    # print('TRAINING USING CNN MODEL.......................')
    # cnn = CNN(
    # title_train_ids,
    # text_train_ids,
    # train_labels,
    # title_val_ids,
    # text_val_ids,
    # val_labels,
    # vocab_size,
    # emb_mat
    # )
    #
    # cnn_model = cnn.trainModel()
    # cnn.testModel(
    # [np.array(title_test_ids), np.array(text_test_ids)], 
    # np.array(test_labels)
    # )
    #
    # print('TRAINING USING BiLSTM MODEL......................')
    # bilstm = BiLSTM(
    # title_train_ids,
    # text_train_ids,
    # train_labels,
    # title_val_ids,
    # text_val_ids,
    # val_labels,
    # vocab_size,
    # emb_mat
    # )
    #
    # bilstm_model = bilstm.trainModel()
    # bilstm.testModel(
    # [np.array(title_test_ids), np.array(text_test_ids)], 
    # np.array(test_labels)
    # )
    #
    # print('TRAINING USING ENSEMBLE CNN + BiLSTM MODEL.................')
    # cnn_bilstm = En_CNN_BILSTM(
    # title_train_ids,
    # text_train_ids,
    # train_labels,
    # title_val_ids,
    # text_val_ids,
    # val_labels,
    # vocab_size,
    # )
    #
    # cnn_bilstm.trainModel()
    # cnn_bilstm.testModel(
    # [np.array(title_test_ids), np.array(text_test_ids)], 
    # np.array(test_labels)
    # )
    #
    # print('TRAINING USING FUSION CNN + BiLSTM MODEL.................')
    # cnn_bilstm = Fu_CNN_BILSTM(
    # title_train_ids,
    # text_train_ids,
    # train_labels,
    # title_val_ids,
    # text_val_ids,
    # val_labels,
    # vocab_size,
    # emb_mat
    # )
    #
    # cnn_bilstm.trainModel()
    # cnn_bilstm.testModel(
    # [np.array(title_test_ids), np.array(text_test_ids)], 
    # np.array(test_labels)
    # )

    # print('TRAINING USING TRANSFORMER MODEL.................')
    # transformer_model = Transformer(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    #     vocab_size,
    #     emb_mat
    # )
    #
    # transformer_model.trainModel()
    # transformer_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )

    # title_train_emb = attachEmbeddingToIds(tokenizer, title_train_ids, emb_mat)
    # title_val_emb = attachEmbeddingToIds(tokenizer, title_val_ids, emb_mat)
    # title_test_emb = attachEmbeddingToIds(tokenizer, title_test_ids, emb_mat)
    #
    # text_train_emb = attachEmbeddingToIds(tokenizer, text_train_ids, emb_mat)
    # text_val_emb = attachEmbeddingToIds(tokenizer, text_val_ids, emb_mat)
    # text_test_emb = attachEmbeddingToIds(tokenizer, text_test_ids, emb_mat)

    # print('TRAINING USING SGD MODEL.................')
    # sgd_model = SGD(
    #     title_train_emb,
    #     text_train_emb,
    #     train_labels,
    #     title_val_emb,
    #     text_val_emb,
    #     val_labels,
    # )
    #
    # sgd_model.trainModel()
    # sgd_model.testModel(
    #     [np.array(title_test_emb), np.array(text_test_emb)], 
    #     np.array(test_labels)
    # )
    #
    # print('TRAINING USING KNN MODEL.................')
    # knn_model = KNN(
    #     title_train_emb,
    #     text_train_emb,
    #     train_labels,
    #     title_val_emb,
    #     text_val_emb,
    #     val_labels,
    # )
    #
    # knn_model.trainModel()
    # knn_model.testModel(
    #     [np.array(title_test_emb), np.array(text_test_emb)], 
    #     np.array(test_labels)
    # )

    # print('TRAINING USING LOGISTIC REGRESSION MODEL.................')
    # lg_model = LOGISTIC_REGRESSION(
    #     title_train_emb,
    #     text_train_emb,
    #     train_labels,
    #     title_val_emb,
    #     text_val_emb,
    #     val_labels,
    # )
    #
    # lg_model.trainModel()
    # lg_model.testModel(
    #     [np.array(title_test_emb), np.array(text_test_emb)], 
    #     np.array(test_labels)
    # )
    #
    # print('TRAINING USING DECISION FOREST MODEL.................')
    # df_model = DECISION_FOREST(
    #     title_train_emb,
    #     text_train_emb,
    #     train_labels,
    #     title_val_emb,
    #     text_val_emb,
    #     val_labels,
    # )
    #
    # df_model.trainModel()
    # df_model.testModel(
    #     [np.array(title_test_emb), np.array(text_test_emb)], 
    #     np.array(test_labels)
    # )

    print('TRAINING USING GRU MODEL.................')
    gru_model = GRU(
        title_train_ids,
        text_train_ids,
        train_labels,
        title_val_ids,
        text_val_ids,
        val_labels,
        vocab_size,
        emb_mat
    )

    gru_model.trainModel()
    gru_model.testModel(
        [np.array(title_test_ids), np.array(text_test_ids)], 
        np.array(test_labels)
    )
    #
    # print('TRAINING USING BIGRU MODEL.................')
    # bigru_model = BIGRU(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    #     vocab_size,
    #     emb_mat
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
    #
    # print('TRAINING USING FUSION CNN + BIGRU MODEL.................')
    # fusion_cnn_bigru_model = Fusion_CNN_BIGRU(
    #     title_train_ids,
    #     text_train_ids,
    #     train_labels,
    #     title_val_ids,
    #     text_val_ids,
    #     val_labels,
    #     vocab_size,
    #     emb_mat
    # )
    #
    # fusion_cnn_bigru_model.trainModel()
    # fusion_cnn_bigru_model.testModel(
    #     [np.array(title_test_ids), np.array(text_test_ids)], 
    #     np.array(test_labels)
    # )
    #
    # print('TRAINING DONE.............................')
