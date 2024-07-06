from loadModel import getModel
from keras.layers import Concatenate
import tensorflow as tf
import numpy as np
from phobert_we.constants import *
from glove_we.gloveEmbedding import getFeature_ML, getFeature_DL
from phobert_we.phoBERTEmbedding import getFeaturePrediction

print('LOADING MODELS >>>>>>>>>>>>>>>')
model = getModel(PATH + MODEL + ENSEMBLE_CNN_BILSTM)

def getRatingFromModel(content):
    content = getFeaturePrediction(content)

    return model.predict(content)
