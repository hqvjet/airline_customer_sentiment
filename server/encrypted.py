from vncorenlp import VnCoreNLP
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle as pkl
import re
import pandas as pd
import numpy as np
from fairseq.data.encoders.fastbpe import fastBPE
import argparse
from fairseq.data import Dictionary
from constants import *

rdr = VnCoreNLP('../glove/resources/' + "vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
with open('../glove/resources/glove/models/' + 'TOKENIZER.pkl', 'rb') as pkl_file:
    tokenizer = pkl.load(pkl_file)
stopword = pd.read_csv('../glove/resources/' + 'stopword.csv')
stopword = [word for word in stopword['stopword']]

parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default='../phobert/resources/' + "PhoBERT_large_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)

vocab = Dictionary()
vocab.add_from_file('../phobert/resources/' + "PhoBERT_large_transformers/dict.txt")
    
def normalizeSentence(sentence):
    RE_EMOJI = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
    sentence = RE_EMOJI.sub(r'', sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(r'\b\d+[\.,/]\s*', '', sentence)

    for word in stopword:
        sentence = sentence.replace(' ' + word + ' ', ' ')
    
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence.lower().strip()

def getIDS(text, emb_tech):
    text = [normalizeSentence(' '.join(' '.join(i) for i in rdr.tokenize(text)))]
    print(text)
    
    if emb_tech == GLOVE_METHOD:
        ids = tokenizer.texts_to_sequences(np.array(text))
        ids = pad_sequences(ids, padding='post', maxlen=200)

    elif emb_tech == PHOBERT_METHOD:
        words = '<s>' + bpe.encode(text) + '</s>'
        ids = vocab.encode_line(words, append_eos=True, add_if_not_exist=False).long().tolist()

    return ids