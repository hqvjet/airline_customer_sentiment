from loadModel import getModel
from keras.layers import Concatenate
import tensorflow as tf
import numpy as np
from constants import *
from phobert_we.phoBERTEmbedding import getFeaturePrediction

phobert_model = getModel(PHOBERT_PATH + ENSEMBLE_CNN_BILSTM_MODEL)

def getRatingFromModel(title, content):
    title = getFeaturePrediction(title)
    content = getFeaturePrediction(content)

    input = [np.array(title), np.array(content)]

    y_pred = phobert_model.predict(input)
       
    print(y_pred)

    return y_pred
