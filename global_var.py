from keras.layers import LSTM, Dense, Embedding, Reshape, Concatenate, Bidirectional
from util import *
from pickle import load


# prepare english tokenizer
with open('eng_tokenizer.pickle', 'rb') as handle:
    eng_tokenizer = load(handle)
eng_vocab_size = len(eng_tokenizer.word_index) + 1
with open('eng_length.pickle', 'rb') as handle:
    eng_length = load(handle)
Ty = eng_length
vocab_tar_size = eng_vocab_size

# prepare german tokenizer
with open('ger_tokenizer.pickle', 'rb') as handle:
    ger_tokenizer = load(handle)
ger_vocab_size = len(ger_tokenizer.word_index) + 1
with open('ger_length.pickle', 'rb') as handle:
    ger_length = load(handle)
Tx = ger_length
vocab_inp_size = ger_vocab_size

n_a = 512
n_s = n_a * 2
embedding_dim = 256
decoder_cell = LSTM(n_s, return_state = True)
output_layer = Dense(vocab_tar_size, activation='softmax')
decoder_embedding = Embedding(vocab_tar_size, embedding_dim, input_length=1)
encoder_embedding = Embedding(vocab_inp_size, embedding_dim, input_length=Tx)
reshape_layer = Reshape((1,))
enc_concat = Concatenate(axis=-1)
encoder_layer = Bidirectional(LSTM(n_a, return_sequences=True, return_state = True))

# load datasets
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')
# prepare training data
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY_oh = encode_output(trainY, eng_vocab_size) #converted to one hot
# prepare validation data
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY_oh = encode_output(testY, eng_vocab_size) #converted to one hot
