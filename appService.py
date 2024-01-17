from loadModel import getModel
from phoBERTEmbedding import getIDS
import numpy as np

model = getModel()

def getRatingFromModel(title, text):
    title_ids = getIDS(title)
    text_ids = getIDS(text)

    y_pred = model.predict([np.array(title_ids), np.array(text_ids)])
    print(y_pred)

    return 1
