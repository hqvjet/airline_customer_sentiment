from loadModel import getModel
from encrypted import getIDS
import numpy as np
from constants import *

print('LOADING MODELS >>>>>>>>>>>>>>>')
glove_cnn = getModel(GLOVE_PATH + CNN_MODEL)
glove_lstm = getModel(GLOVE_PATH + LSTM_MODEL)
glove_bilstm = getModel(GLOVE_PATH + BILSTM_MODEL)
glove_ensemble = getModel(GLOVE_PATH + ENSEMBLE_CNN_BILSTM)
glove_fusion = getModel(GLOVE_PATH + FUSION_CNN_BILSTM)

print('LOADING GLOVE MODE DONE..................')
phobert_cnn = getModel(PHOBERT_PATH + CNN_MODEL)
phobert_lstm = getModel(PHOBERT_PATH + LSTM_MODEL)
phobert_bilstm = getModel(PHOBERT_PATH + BILSTM_MODEL)
phobert_ensemble = getModel(PHOBERT_PATH + ENSEMBLE_CNN_BILSTM)
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
        elif model_name == ENSEMBLE_CNN_BILSTM:
            y_pred = glove_ensemble.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == FUSION_CNN_BILSTM:
            y_pred = glove_fusion.predict([np.array(title_ids), np.array(text_ids)])
            
    elif emb_tech == PHOBERT_METHOD:
        if model_name == CNN_MODEL:
            y_pred = glove_cnn.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == LSTM_MODEL:
            y_pred = glove_lstm.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == BILSTM_MODEL:
            y_pred = glove_bilstm.predict([np.array(title_ids), np.array(text_ids)])
        elif model_name == ENSEMBLE_CNN_BILSTM:
            y_pred = glove_ensemble.predict([np.array(title_ids), np.array(text_ids)])
    
    print(y_pred)

    return y_pred