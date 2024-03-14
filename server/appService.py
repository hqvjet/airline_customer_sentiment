from loadModel import getModel
from encrypted import getIDS
import numpy as np

print('LOADING MODELS >>>>>>>>>>>>>>>')
GLOVE_PATH = '../glove/resources/glove/models/'
glove_cnn = getModel(GLOVE_PATH + 'CNN_MODEL.keras')
glove_lstm = getModel(GLOVE_PATH + 'LSTM_MODEL.keras')
glove_bilstm = getModel(GLOVE_PATH + 'BILSTM_MODEL.keras')
glove_ensemble = getModel(GLOVE_PATH + 'ENSEMBLE_CNN_BILSTM_MODEL.keras')
glove_fusion = getModel(GLOVE_PATH + 'FUSION_CNN_BILSTM_MODEL.keras')

def getRatingFromModel(title, content, model_name):
    title_ids = getIDS(title)
    text_ids = getIDS(content)

    if model_name == 'CNN_MODEL.keras':
        y_pred = glove_cnn.predict([np.array(title_ids), np.array(text_ids)])
    elif model_name == 'LSTM_MODEL.keras':
        y_pred = glove_lstm.predict([np.array(title_ids), np.array(text_ids)])
    elif model_name == 'BILSTM_MODEL.keras':
        y_pred = glove_bilstm.predict([np.array(title_ids), np.array(text_ids)])
    elif model_name == 'ENSEMBLE_CNN_BILSTM_MODEL.keras':
        y_pred = glove_ensemble.predict([np.array(title_ids), np.array(text_ids)])
    elif model_name == 'FUSION_CNN_BILSTM_MODEL.keras':
        y_pred = glove_fusion.predict([np.array(title_ids), np.array(text_ids)])
    
    print(y_pred)

    return y_pred
