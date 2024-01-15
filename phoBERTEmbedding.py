# # Download VnCoreNLP-1.1.1.jar & its word segmentation component (i.e. RDRSegmenter) 
# mkdir -p vncorenlp/models/wordsegmenter
# wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
# wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
# wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
# mv VnCoreNLP-1.1.1.jar vncorenlp/ 
# mv vi-vocab vncorenlp/models/wordsegmenter/
# mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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
from Nomarlize import normalizeSentence

# LOAD MODEL AND BPE
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)
rdr = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("/content/drive/My Drive/BERT/SA/PhoBERT_base_transformers/dict.txt")

def getDataIDS(data):
    data_ids = []
    for sentence in data:
        words = '<s>' + bpe.encode(sentence) + '</s>'
        encoded_setence = vocab.encode_line(words, append_eos=True, add_if_not_exist=False).long().tolist()
        data_ids.append(encoded_setence)

    # PAD TO MAXLEN
    data_ids.pad_sequences(data_ids, maxlen=MAX_LEN, dtype='long', value=0, truncating='post', padding='post')

    return torch.tensor(data_ids)

def getMask(data_ids):
    data_masks = []
    for sentence in data_ids:
        mask = [int(token_id > 0) for token_id in sentence]
        data_masks.append(mask)

    return torch.tensor(data_masks)

def getDataset(file_name):
    # GET FROM CSV (ORIGINAL TEXT)
    file = pd.read_csv(PATH + file_name)

    title = file['title'].apply(str)
    text = file['text'].apply(str)
    label = utils.to_categorical(file['rating'] - 1, num_classes=3)

    # NORMALIZE DATASET
    title = [normalizeSentence(sentence) for sentence in title]
    text = [normalizeSentence(sentence) for sentence in text]

    # TOKENIZE DATASET
    title = [rdr.tokenize(sentence) for sentence in title]
    text = [rdr.tokenize(sentence) for sentence in text]

    # MAPPING TO VOCAB
    title_ids = getDataIDS(title)
    text_ids = getDataIDS(text)

    # # CREATE MASK
    # title_masks = getMask(title_ids)
    # text_masks = getMask(text_ids)

    # data = TensorDataset(title_ids, title_masks, label)
    # data_sampler = SequentialSampler(data)
    # dataloader = DataLoader(data, sampler=data_sampler, batch_size=BATCH_SIZE)

    return title_ids, text_ids, label


def usingPhoBERT():
    title_train_ids, text_train_ids, train_labels = getDataset('train.csv')
    title_test_ids, text_test_ids, test_labels = getDataset('test.csv')

    title_train_ids, title_val_ids, text_train_ids, text_val_ids, train_labels, val_labels = train_test_split(title_train_ids, text_train_ids, train_labels)

    return title_train_ids, text_train_ids, train_labels, title_val_ids, text_val_ids, val_labels, title_test_ids, text_test_ids, test_labels
