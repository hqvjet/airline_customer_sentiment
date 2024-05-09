from loadModel import getModel
from encrypted import getIDS
import numpy as np
from constants import *

print('LOADING MODELS >>>>>>>>>>>>>>>')
glove_cnn = getModel(GLOVE_PATH + CNN_MODEL)
glove_lstm = getModel(GLOVE_PATH + LSTM_MODEL)
glove_bilstm = getModel(GLOVE_PATH + BILSTM_MODEL)
glove_gru = getModel(GLOVE_PATH + GRU_MODEL)
glove_bigru = getModel(GLOVE_PATH + BIGRU_MODEL)
glove_ensemble_lstm = getModel(GLOVE_PATH + ENSEMBLE_CNN_BILSTM_MODEL)
glove_ensemble_gru = getModel(GLOVE_PATH + ENSEMBLE_CNN_BIGRU_MODEL)
glove_fusion_lstm = getModel(GLOVE_PATH + FUSION_CNN_BILSTM_MODEL)
glove_fusion_gru = getModel(GLOVE_PATH + FUSION_CNN_BIGRU_MODEL)
glove_transformer = getModel(GLOVE_PATH + TRANSFORMER_MODEL)
glove_sgd = getModel(GLOVE_PATH + SGD_MODEL)
glove_rand_for = getModel(GLOVE_PATH + DECISION_FOREST_MODEL)
glove_log_reg = getModel(GLOVE_PATH + LOGIS_REG_MODEL)

print('LOADING GLOVE MODE DONE..................')
phobert_cnn = getModel(PHOBERT_PATH + CNN_MODEL)
phobert_lstm = getModel(PHOBERT_PATH + LSTM_MODEL)
phobert_bilstm = getModel(PHOBERT_PATH + BILSTM_MODEL)
phobert_gru = getModel(PHOBERT_PATH + GRU_MODEL)
phobert_bigru = getModel(PHOBERT_PATH + BIGRU_MODEL)
phobert_ensemble_lstm = getModel(PHOBERT_PATH + ENSEMBLE_CNN_BILSTM_MODEL)
phobert_ensemble_gru = getModel(PHOBERT_PATH + ENSEMBLE_CNN_BIGRU_MODEL)
phobert_fusion_lstm = getModel(PHOBERT_PATH + FUSION_CNN_BILSTM_MODEL)
phobert_fusion_gru = getModel(PHOBERT_PATH + FUSION_CNN_BIGRU_MODEL)
phobert_transformer = getModel(PHOBERT_PATH + TRANSFORMER_MODEL)
phobert_sgd = getModel(PHOBERT_PATH + SGD_MODEL)
phobert_rand_for = getModel(PHOBERT_PATH + DECISION_FOREST_MODEL)
phobert_log_reg = getModel(PHOBERT_PATH + LOGIS_REG_MODEL)
print('LOADING PHOBERT MODEL DONE ................')

def getRatingFromModel(title, content, model_name, emb_tech):
    title_ids = getIDS(title, emb_tech)
    text_ids = getIDS(content, emb_tech)

    if emb_tech == GLOVE_METHOD:
        if model_name == CNN_MODEL:
            y_pred = glove_cnn.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == LSTM_MODEL:
            y_pred = glove_lstm.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == BILSTM_MODEL:
            y_pred = glove_bilstm.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == ENSEMBLE_CNN_BILSTM_MODEL:
            y_pred = glove_ensemble_lstm.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == FUSION_CNN_BILSTM_MODEL:
            y_pred = glove_fusion_lstm.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == ENSEMBLE_CNN_BIGRU_MODEL:
            y_pred = glove_ensemble_gru.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == FUSION_CNN_BIGRU_MODEL:
            y_pred = glove_fusion_gru.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == TRANSFORMER_MODEL:
            y_pred = glove_transformer.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == SGD_MODEL:
            y_pred = glove_sgd.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == DECISION_FOREST_MODEL:
            y_pred = glove_rand_for.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == LOGIS_REG_MODEL:
            y_pred = glove_log_reg.predict([np.array(title_ids), np.array(text_ids)])
            
    elif emb_tech == PHOBERT_METHOD:
        if model_name == CNN_MODEL:
            y_pred = phobert_cnn.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == LSTM_MODEL:
            y_pred = phobert_lstm.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == BILSTM_MODEL:
            y_pred = phobert_bilstm.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == GRU_MODEL:
            y_pred = phobert_gru.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == BIGRU_MODEL:
            y_pred = phobert_bigru.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == ENSEMBLE_CNN_BILSTM_MODEL:
            y_pred = phobert_ensemble.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == ENSEMBLE_CNN_BIGRU_MODEL:
            y_pred = phobert_ensemble_gru.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == FUSION_CNN_BILSTM_MODEL:
            y_pred = phobert_fusion_lstm.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == FUSION_CNN_BIGRU_MODEL:
            y_pred = phobert_cnn.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == TRANSFORMER_MODEL:
            y_pred = phobert_transformer.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == SGD_MODEL:
            y_pred = phobert_sgd.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == DECISION_FOREST_MODEL:
            y_pred = phobert_rand_for.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == LOGIS_REG_MODEL:
            y_pred = phobert_log_reg.predict([np.array(title_ids), np.array(text_ids)])
    
    print(y_pred)

    return y_pred
