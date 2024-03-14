from keras.models import load_model

# USING TRAINED BILSTM MODEL
def getModel(path):
    model = load_model(path)
    return model