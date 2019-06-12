
This is a Neural Machine Translation (NMT) model that translates a sentence of one language to another, given the traning sets
containing the text pair of one language and the translated language.
The architecture is encoder-decoder with attention mechanism.
It uses keras.

Datasets:
The whole dataset is first split into input language and target language to get
the the input and target languages total vocabulary counts and maximum input,
target languages' sentences' max length by words counts, and tokenizers that convert sentences to sequences of integers representation and convert them back.

Architecture:
The attention mechanism is used to tell the decoder to which part of the input sequence
to pay more attention.  The complete input sequence needs to finish to compute
the attention vector for each decoder time step.  Therefore Bidirectional LSTM is used in
the encoder to also improve the translation since the whole input sequence has to complete first.
 However, this makes the output sequence generation more complicated, since we don't know all the output seqence in advance;
instead we generate them one at a time using x⟨t⟩=y⟨t−1⟩. a for-loop to iterate over the decoder time steps is needed and need to implement it manually.
For how this can be done using keras (it is kind of tricky), please see my other projects, the music generation.
This also make the input sequence padding to max necessary because Tx needs to be fixed.
The training model (model) is slightly different than the reference model (ref_model) because in training, teacher forcing is used but in reference model sampling is used.
This also make the RNN layers (LSTM used in this project) variables in model to be shared for ref_model necessary.

Notation:
x<t> is the input at encoder timestep t.
x<t'> t' is the decoder timestep.
n_a is the number of hidden state/unit for encoder
n_s is the number of hidden state/unit for decoder
c is the memory cell state for decoder
s is the hidden unit of decoder LSTM
a is the hidden unit of encoder


The choice of n_a and n_s (decoder hidden state) is pretty arbitrary.  I used 1024 as in the tensorflow tutorial.  I tried other values and didn't seem to make much difference.
n_s need to be twice of n_a because encoder LSTM is a bi-LSTM.
Tx need to be fixed because to the way attention vector is calculated here, that by stacking encoder hidden states with alpha, 
if Tx is different for each training example, then each context<Ty> will have
different dimensions, and won't be able to stack with s<Ty> (decoder hidden state), since s<Ty> need
to have same dimension for each time step Ty in decoder.
the architecture is no longer a "true" encoder-decoder model because of the introduction
of attention mechanism.
n_a is used to define encoder Bi-LSTM cell in the model:
a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
n_s is used to define decoder LSTM cell:
decoder_LSTM_cell = LSTM(n_s, return_state = True)


the return_state = True parameter is neccesary for post attention LSTM layer
because we also need the c (cell state) as well as s.

References:
Deep Learning for Natural Language Processing by Jason Brownlee
https://www.analyticsvidhya.com/blog/2019/01/neural-machine-translation-keras/
https://www.coursera.org/learn/nlp-sequence-models/notebook/npjGi/neural-machine-translation-with-attention
https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/

to run:
clean.py
split.py
train.py
eval.py

to change the size of training samples:
change n_sentences in split.py, and run split.py again to generate the dataset files.
for 5000, training runs about 30 minutes on gpu with 8g memory.
