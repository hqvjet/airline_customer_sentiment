from keras.models import load_model
from constants import *

# USING TRAINED BILSTM MODEL
def getModel():
    model = load_model(PATH + BILSTM_MODEL)
    return model