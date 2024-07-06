from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from phobert_we.constants import *
from tensorflow.keras import utils
from phobert_we.Nomarlize import *

def usePhoBERT():
    print('READING PhoBERT FEATURES.................................')
    text_train = np.load(PATH + MODEL + PHOBERT_FEATURES_TEXT)
    print(f'Shape: {text_train.shape}')

    print('READING DATASET FROM FILE................................')
    labels = pd.read_csv('hf://datasets/florentgbelidji/car-reviews/train_car.csv')[:text_train.shape[0]]['Rating']
    labels = pd.Series([arrayToStatus(label) for label in labels])
    print(labels.info)

    # labels = pd.Series([status for status in file['Rating'].apply(int)]) 
    labels = utils.to_categorical(labels - 1, num_classes=2)

    # SPLIT DATASET
    # text_train, text_test, train_labels, test_labels = train_test_split(text_train, labels, test_size=0.2)
    # text_train, text_val, train_labels, val_labels = train_test_split(text_train, train_labels, test_size=0.2)

    print('DATA SPLIT DONE.................')
    l_t = text_train.shape[0]
    l_l = labels.shape[0]
    return text_train[:int(0.7*l_t)], labels[:int(0.7*l_l)], text_train[int(0.7*l_t):int(0.8*l_t)], labels[int(0.7*l_l):int(0.8*l_l)], text_train[int(0.8*l_t):], labels[int(0.8*l_l):]
