# # Download VnCoreNLP-1.1.1.jar & its word segmentation component (i.e. RDRSegmenter) 
# mkdir -p vncorenlp/models/wordsegmenter
# wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
# wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
# wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
# mv VnCoreNLP-1.1.1.jar vncorenlp/ 
# mv vi-vocab vncorenlp/models/wordsegmenter/
# mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import utils
from vncorenlp import VnCoreNLP
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse
import pandas as pd
import torch
from constants import *
from Nomarlize import normalizeSentence, statusToNumber
import numpy as np
from transformers import AutoModel, TFAutoModel, AutoTokenizer
# LOAD MODEL AND BPE
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default=PATH + "PhoBERT_large_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)
rdr = VnCoreNLP(PATH + "vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
# Load the dictionary
vocab = Dictionary()
vocab.add_from_file(PATH + "PhoBERT_large_transformers/dict.txt")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

def test():
    phobert, tokenizer = loadPhobert()
    data = ['Tôi là sinh viên trường đại học bách khoa hà nội', 'Nhiều khi anh mong được một lần nói ra hết tất cả thay vì']
    print(torch.tensor([tokenizer.encode(data[0])]))

    data = [normalizeSentence(' '.join(' '.join(i) for i in rdr.tokenize(sentence))) for sentence in data]
    print(data)
    print(bpe.encode(data[0]))
    print(tokenizer.encode(data[0]))
    print(vocab.encode_line(data[0], append_eos=True, add_if_not_exist=False).long().tolist())
    

def loadPhobert():
    print('Loading PhoBERT model............')
    phobert = TFAutoModel.from_pretrained("vinai/phobert-large")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")

    return phobert, tokenizer
    
def getDataIDS(data):
    data_ids = []
    for sentence in data:
        encoded_setence = tokenizer.encode(sentence)
        data_ids.append(encoded_setence)
    # PAD TO MAXLEN
    data_ids = pad_sequences(data_ids, maxlen=MAX_LEN, dtype='long', value=1, truncating='post', padding='post')

    return data_ids

def getAttentionMask(data_ids):
    print('GETTING ATTENTION MASK..................................')
    data_masks = np.where(data_ids == 1, 0, 1)
    print(data_masks.shape)
    print(data_masks[0])

    return data_masks

def prepareData(title, text):
    # TOKENIZE DATASET
    print('TOKENIZING DATASET.......................................')
    title = [normalizeSentence(' '.join(' '.join(i) for i in rdr.tokenize(sentence))) for sentence in title]
    text = [normalizeSentence(' '.join(' '.join(i) for i in rdr.tokenize(sentence))) for sentence in text]

    # MAPPING TO VOCAB
    print('MAPPING AND PADDING DATASET..............................')
    title_ids = getDataIDS(title)
    text_ids = getDataIDS(text)

    # # CREATE MASK
    # title_masks = getMask(title_ids)
    # text_masks = getMask(text_ids)

    # data = TensorDataset(title_ids, title_masks, label)
    # data_sampler = SequentialSampler(data)
    # dataloader = DataLoader(data, sampler=data_sampler, batch_size=BATCH_SIZE)

    return title_ids, text_ids

def getDataset(file_name):
    # GET FROM CSV (ORIGINAL TEXT)
    print('READING DATASET FROM FILE................................')
    file = pd.read_csv(PATH + file_name)

    title = file['Title'].apply(str)
    text = file['Content'].apply(str)

    # GET LABELS
    label = pd.Series([status for status in file['Rating'].apply(int)])
    label = utils.to_categorical(label - 1, num_classes=3)

    return title, text, label

def getFeature(ids):

    print('LOADING PHOBERT MODEL......................')
    phobert = TFAutoModel.from_pretrained("vinai/phobert-large")
    # phobert2 = AutoModel.from_pretrained("vinai/phobert-large")
    print('EXTRACTING FEATURES........................')
    output = phobert(input_ids=ids, attention_mask=getAttentionMask(ids))
    # with torch.no_grad():
        # last_hidden_states = phobert2(input_ids=torch.tensor(ids), attention_mask=torch.tensor(getAttentionMask(ids)))

    features = output.last_hidden_state.numpy()
    # features = last_hidden_states[0][:, 0, :].numpy()
    print(features.shape)

    return features

def usingPhoBERT():
    title, text, labels = getDataset('data.csv')

    # IDS input
    title_ids, text_ids = prepareData(title, text)

    # GET features
    title_features = getFeature(title_ids)
    text_features = getFeature(text_ids)
    
    print(title_features)
    print(type(title_features))
    # SPLIT DATASET
    title_train, title_test, text_train, text_test, train_labels, test_labels = train_test_split(title_features, text_features, labels, test_size=0.1, random_state=0)
    title_train, title_val, text_train, text_val, train_labels, val_labels = train_test_split(title_train, text_train, train_labels, test_size=0.1, random_state=0)

    return title_train, text_train, train_labels, title_val, text_val, val_labels, title_test, text_test, test_labels, len(vocab)

def getIDS(sentence):
    sentence = normalizeSentence(sentence)
    sentence = rdr.tokenize(sentence)
    sentence = sentence[0]  
    print(sentence)
    sentence = [' '.join(sentence)]
    sentence_ids = getDataIDS(sentence)

    return sentence_ids
