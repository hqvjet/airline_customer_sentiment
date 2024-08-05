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
from phobert_we.constants import *
from phobert_we.Nomarlize import normalizeSentence, statusToNumber
import numpy as np
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import math
# LOAD MODEL AND BPE
# parser = argparse.ArgumentParser()
# parser.add_argument('--bpe-codes', 
#     default=PATH + "PhoBERT_large_transformers/bpe.codes",
#     required=False,
#     type=str,
#     help='path to fastBPE BPE'
# )
# args, unknown = parser.parse_known_args()
# bpe = fastBPE(args)
# Load the dictionary
rdr = VnCoreNLP("tools/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
# vocab = Dictionary()
# vocab.add_from_file(PATH + "PhoBERT_large_transformers/dict.txt")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def getDataIDS(data):
    data_ids = []
    for sentence in data:
        encoded_setence = tokenizer.encode(sentence)
        data_ids.append(encoded_setence)
    # PAD TO MAXLEN
    data_ids = pad_sequences(data_ids, maxlen=MAX_LEN, dtype='long', value=1, truncating='post', padding='post')

    return data_ids

def getAttentionMask(data_ids):
    data_masks = np.where(data_ids == 1, 0, 1)

    return data_masks

def prepareData(text):
    # MAPPING TO VOCAB
    print('MAPPING AND PADDING DATASET..............................')
    text_ids = getDataIDS(text)

    return text_ids

def getDataset(file_name):
    # GET FROM CSV (ORIGINAL TEXT)
    print('READING DATASET FROM FILE................................')
    file = pd.read_csv(file_name)

    text = file['Review'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.replace(r'\r', ' ', regex=True).str.lower()[:10000]
    
    print(text.info)
    return text

def getFeature(ids):

    print('LOADING PHOBERT MODEL......................')
    phobert = TFBertModel.from_pretrained("bert-base-uncased", from_pt=True)
    features = []

    print('EXTRACTING FEATURES........................')

    # dataset = tf.data.Dataset.from_tensor_slices(ids)
    # dataset = dataset.batch(PHOBERT_BATCH_SIZE)

    # for ids_element in dataset:
    #     output = phobert(input_ids=ids_element.numpy())
    #
    #     features.append(output[0].numpy())
    # print(features[-1].shape)

    num_batches = math.ceil(ids.shape[0] // PHOBERT_BATCH_SIZE)

    for batch in range(num_batches):
        batch_element = ids[batch * PHOBERT_BATCH_SIZE : min((batch + 1) * PHOBERT_BATCH_SIZE, ids.shape[0])]
        output = phobert(input_ids=batch_element, attention_mask=getAttentionMask(batch_element))
        features.append(output.last_hidden_state)

        print(f'Batch number {batch + 1} finished, Shape: {output.last_hidden_state.shape}')
    # output = phobert(input_ids=ids)
    features = tf.concat(features, axis=0)
    # features = features.reshape(len(ids), MAX_LEN, 768)

    # output = phobert(input_ids=ids, attention_mask=getAttentionMask(ids))
    # features = output.last_hidden_state.numpy()
    print(features.shape)

    return features

def extractFeatures():
    text = getDataset('hf://datasets/florentgbelidji/car-reviews/train_car.csv')

    # IDS input
    text_ids = prepareData(text)

    # GET features
    text_features = getFeature(text_ids)

    np.save('phobert_we/resources/phobert/bwd/features.npy', text_features)

    print('SAVED.....................')

    # SPLIT DATASET
    # title_train, title_test, text_train, text_test, train_labels, test_labels = train_test_split(title_train, text_train, labels, test_size=0.1)
    # title_train, title_val, text_train, text_val, train_labels, val_labels = train_test_split(title_train, text_train, train_labels, test_size=0.1)
    # print('DATA SPLIT DONE.................')
    #
    # return title_train, text_train, train_labels, title_val, text_val, val_labels, title_test, text_test, test_labels

phobert = TFBertModel.from_pretrained("bert-base-uncased", from_pt=True)
def getFeaturePrediction(sentence):
    sentence = normalizeSentence(' '.join(' '.join(i) for i in rdr.tokenize(sentence)))
    ids = getDataIDS([sentence])
    features = []
    output = phobert(input_ids=ids, attention_mask=getAttentionMask(ids))
    features = np.array(output.last_hidden_state)
    return features
