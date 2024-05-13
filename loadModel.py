from keras.models import load_model
import keras
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer
from constants import *
import joblib

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# USING TRAINED BILSTM MODEL
def getModel(path):
    if path == GLOVE_PATH + SGD_MODEL or path == GLOVE_PATH + LOGIS_REG_MODEL or path == GLOVE_PATH + DECISION_FOREST_MODEL or path == PHOBERT_PATH + SGD_MODEL or path == PHOBERT_PATH + LOGIS_REG_MODEL or path == PHOBERT_PATH + DECISION_FOREST_MODEL:
        model = joblib.load(path)
    elif path != GLOVE_PATH + TRANSFORMER_MODEL and path != PHOBERT_PATH + TRANSFORMER_MODEL:
        model = load_model(path)
    else:
        model = load_model(path, custom_objects={'TransformerBlock': TransformerBlock})

    return model
