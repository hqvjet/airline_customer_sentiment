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

def getDataIDS(data):
    data_ids = []
    for sentence in data:
        words = '<s>' + bpe.encode(sentence) + '</s>'
        encoded_setence = vocab.encode_line(words, append_eos=True, add_if_not_exist=False).long().tolist()
        data_ids.append(encoded_setence)
    # PAD TO MAXLEN
    data_ids = pad_sequences(data_ids, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')

    return torch.tensor(data_ids)

def getMask(data_ids):
    data_masks = []
    for sentence in data_ids:
        mask = [int(token_id > 0) for token_id in sentence]
        data_masks.append(mask)

    return torch.tensor(data_masks)

def prepareData(title, text):
    # NORMALIZE DATASET
    title = [normalizeSentence(sentence) for sentence in title]
    text = [normalizeSentence(sentence) for sentence in text]

    # TOKENIZE DATASET
    print('TOKENIZING DATASET.......................................')
    title = [rdr.tokenize(sentence) for sentence in title]
    text = [rdr.tokenize(sentence) for sentence in text]
    title = [[' '.join(word) for word in sentence] for sentence in title]
    text = [[' '.join(word) for word in sentence] for sentence in text]
    title = [' '.join(sentence) for sentence in title]
    text = [' '.join(sentence) for sentence in text]

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

    title = file['title'].apply(str)
    text = file['text'].apply(str)

    # GET LABELS
    label = pd.Series([statusToNumber(status) for status in file['rating'].apply(str)])
    label = utils.to_categorical(label - 1, num_classes=3)

    return title, text, label

def usingPhoBERT():
    title_train, text_train, train_labels = getDataset('train.csv')
    title_test, text_test, test_labels = getDataset('test.csv')

    title_train, title_val, text_train, text_val, train_labels, val_labels = train_test_split(title_train, text_train, train_labels, test_size=0.1)

    title_train_ids, text_train_ids = prepareData(title_train, text_train)
    title_val_ids, text_val_ids = prepareData(title_val, text_val)
    title_test_ids, text_test_ids = prepareData(title_test, text_test)

    return title_train_ids, text_train_ids, train_labels, title_val_ids, text_val_ids, val_labels, title_test_ids, text_test_ids, test_labels, len(vocab)

def getIDS(sentence):
    sentence = normalizeSentence(sentence)
    sentence = [rdr.tokenize(sentence)]
    sentence_ids = getDataIDS(sentence)

    return sentence_ids