from glove import Corpus, Glove
import pandas as pd
import numpy as np
from tensorflow.keras import utils
from vncorenlp import VnCoreNLP
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

rdr = VnCoreNLP('../glove/resources/' + "vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
tokenizer = Tokenizer().load('../glove/resources/glove/models/tokenizer.keras')

def getIDS(text):
    text = ' '.join(rdr.tokenize(text))
    print(text)

    ids = tokenizer.texts_to_sequences(text)
    ids = pad_sequences(ids, padding='post', maxlen=200)

    return ids