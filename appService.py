from loadModel import getModel
from keras.layers import Concatenate
import tensorflow as tf
import numpy as np
from phobert_we.constants import *
from phobert_we.phoBERTEmbedding import getFeaturePrediction

print('LOADING MODELS >>>>>>>>>>>>>>>')
model = getModel('phobert_we/resources/phobert/bwd/model.keras')

def getRatingFromModel(content):
    content = getFeaturePrediction(content)
    print(content.shape)

    return model.predict(content)
