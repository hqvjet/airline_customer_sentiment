from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from constants import *
from tensorflow.keras import utils

def usePhoBERT():
    print('READING PhoBERT FEATURES.................................')
    title_train = np.load(PATH + MODEL + PHOBERT_FEATURES_TITLE)
    text_train = np.load(PATH + MODEL + PHOBERT_FEATURES_TEXT)

    print('READING DATASET FROM FILE................................')
    file = pd.read_csv(PATH + 'data.csv')
    # GET LABELS

    labels = pd.Series([status for status in file['Rating'].apply(int)])
    labels = utils.to_categorical(labels - 1, num_classes=3)

    threshold = (8000 // PHOBERT_BATCH_SIZE) * PHOBERT_BATCH_SIZE
    labels = labels[:threshold]

    # SPLIT DATASET
    title_train, title_test, text_train, text_test, train_labels, test_labels = train_test_split(title_train, text_train, labels, test_size=0.1)
    title_train, title_val, text_train, text_val, train_labels, val_labels = train_test_split(title_train, text_train, train_labels, test_size=0.1)
    print('DATA SPLIT DONE.................')

    return title_train, text_train, train_labels, title_val, text_val, val_labels, title_test, text_test, test_labels


