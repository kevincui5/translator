from util import *
from global_var import *
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda, Reshape
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
from numpy.random import shuffle
import keras.preprocessing.text
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Embedding

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # customed softmax
dotor = Dot(axes = 1)


# implementation from coursera RNN course week 3 project assignment
def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    
    return context

#define inference model for evaluation
def inference_model(vocab_inp_size, vocab_tar_size, Tx, Ty, n_s, embedding_dim):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    embedding_dim -- embedding layer output size
    n_s -- hidden state size of the post-attention LSTM
    vocab_inp_size -- size of the python dictionary "vocab_inp_size"
    vocab_tar_size -- size of the python dictionary "vocab_tar_size"

    Returns:
    inference_model -- Keras inference model instance
    """
    
    # Define the inputs of your model with a shape (Tx, vocab_inp_size)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X0 = Input(shape=(Tx, ),name='X')
    # (m,Tx)
    X = encoder_embedding(X0)
    # (m,Tx,embedding_dim)
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    outputs = []
    decoder_X0 = Input(shape=(1,))
    decoder_X = decoder_X0
    #shape=(m, 1)
    #shape is not (m, Ty) because we manually iterate Ty timesteps
    
    #Define encoder as Bi-LSTM
    encoder_output, forward_h, forward_c, backward_h, backward_c = encoder_layer(X)
    encoder_hidden = Concatenate(axis=-1)([forward_h, backward_h])
    encoder_cell = Concatenate(axis=-1)([forward_c, backward_c])
    
    s = encoder_hidden
    c = encoder_cell
    for t in range(Ty):
        # one step of the attention mechanism to get back the context vector at step t
        context = one_step_attention(encoder_output, encoder_hidden)
        #(m, 1, n_s)
        decoder_X = decoder_embedding(decoder_X)
        #(m,1) - (m,1,embedding_dim)
        decoder_inputs = enc_concat([decoder_X, context])
        #shape--(m, 1, n_s+embedding_dim)
        # Apply the post-attention LSTM cell to the "context" vector.
        #initial_state = [hidden state, cell state]
        decoder_X, s, c = decoder_cell(decoder_inputs, initial_state = [s, c])
        #decoder_X.shape--(m,n_s)
        # Step 2.C: Apply Dense layer to the hidden state output of the decoder LSTM
        decoder_X = output_layer(decoder_X)
        #(m,vocab_tar_size)
        out = decoder_X
        #trick to add a dimension of 1 to tensor
        decoder_X = RepeatVector(1)(decoder_X)
        decoder_X = Lambda(lambda x: K.argmax(x))(decoder_X) #sampling
        #shape--(m,1) so that it can fit embedding layer
        outputs.append(out)
    
    model = Model([X0, s0, c0, decoder_X0], outputs)
    return model


inf_model = inference_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, n_s, embedding_dim)
inf_model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='categorical_crossentropy')
#there is issues serializing the model architecture, so only save weights
inf_model.load_weights('inf_model_wts.h5')

# evaluate the skill of the model
def translate(model, inp_tokenizer, tar_tokenizer, length, text):
    source = np.expand_dims(text, axis=0) # add a dimension of 1 as a single sample for prediction of one sentence
    source = encode_sequences(inp_tokenizer, length, source)
    # translate encoded source text
    
    inputs = [source, np.zeros((source.shape[0],n_s)), np.zeros((source.shape[0],n_s)), np.zeros((source.shape[0], 1))]
    translation = predict_sequence(model, tar_tokenizer, inputs)
    print('src=[%s], predicted=[%s]' % (text, translation))

 
#inf_model = load_model('inf_model.h5')
translate(inf_model, ger_tokenizer, eng_tokenizer, ger_length, 'nimm dir davon')
