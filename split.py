# -*- coding: utf-8 -*-

from pickle import load, HIGHEST_PROTOCOL
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
from util import *

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('english-german.pkl')

# reduce dataset size
#n_sentences = len(raw_dataset)
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
ratio = 0.95
train_size = int(ratio*n_sentences)
train, test = dataset[:train_size], dataset[train_size:]
# save
#save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')

eng_tokenizer = create_tokenizer(dataset[:, 0])
with open('eng_tokenizer.pickle', 'wb') as handle:
    dump(eng_tokenizer, handle, protocol=HIGHEST_PROTOCOL)
ger_tokenizer = create_tokenizer(dataset[:, 1])
with open('ger_tokenizer.pickle', 'wb') as handle:
    dump(ger_tokenizer, handle, protocol=HIGHEST_PROTOCOL)
eng_length = max_length(dataset[:, 0])
with open('eng_length.pickle', 'wb') as handle:
    dump(eng_length, handle, protocol=HIGHEST_PROTOCOL)
ger_length = max_length(dataset[:, 1])
with open('ger_length.pickle', 'wb') as handle:
    dump(ger_length, handle, protocol=HIGHEST_PROTOCOL)