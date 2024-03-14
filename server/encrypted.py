from vncorenlp import VnCoreNLP
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle as pkl
import re
import pandas as pd

rdr = VnCoreNLP('../glove/resources/' + "vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
with open('../glove/resources/glove/models/' + 'TOKENIZER.pkl', 'rb') as pkl_file:
    tokenizer = pkl.load(pkl_file)
stopword = pd.read_csv('../glove/resources/' + 'stopword.csv')
stopword = [word for word in stopword['stopword']]
    
def normalizeSentence(sentence):
    RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    sentence = RE_EMOJI.sub(r'', sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(r'\b\d+[\.,/]\s*', '', sentence)

    for word in stopword:
        sentence = sentence.replace(' ' + word + ' ', ' ')
    
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.lower().strip()

def getIDS(text):
    text = normalizeSentence(' '.join(' '.join(i) for i in rdr.tokenize(text)))
    print(text)

    ids = tokenizer.texts_to_sequences(text)
    ids = pad_sequences(ids, padding='post', maxlen=200)

    return ids