import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from transformers import AutoTokenizer, AutoModel
from underthesea import word_tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from constants import *
from Nomarlize import normalizeSentence

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phoBERT = AutoModel.from_pretrained("vinai/phobert-base")

def getPhoBERTFeatures(data):

    # PREPARE DATA

    # def getData(file_name):
    #     file = pd.read_csv(PATH + file_name)

    #     text = pd.Series([normalizeSentence(sent) for sent in file['text'].apply(str)])

    #     return text

    # # GET DATA AND STOPWORD
    # data = getData('train.csv')

    data = pd.Series([word_tokenize(sentence, format='text') for sentence in data])

    encoded_sequences = [tokenizer.encode(encode, max_length=MAX_LEN, pad_to_max_length=True) for encode in data] # HIGH RISK

    padded = np.array(encoded_sequences)
    print('padded:', padded[1])
    print('len padded:', padded.shape)

    #get attention mask ( 0: not has word, 1: has word)
    attention_mask = np.where(padded==0, 0, 1)
    # print('attention mask:', attention_mask[1])

    # Convert input to tensor
    padded = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    # Train model
    with torch.no_grad():
        print('TRAINNING PHOBERT MODEL')
        last_hidden_states = phoBERT(input_ids=padded, attention_mask=attention_mask)
    #     print('last hidden states:', last_hidden_states)
    features = last_hidden_states[0][:,0,:].numpy()
    print('PHOBERT MODEL DONE')
    return features