# -*- coding: utf-8 -*-
from util import *
from global_var import *
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda, Reshape
import keras.backend as K
import numpy as np
from keras import optimizers
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping

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

def model(vocab_inp_size, vocab_tar_size, Tx, Ty, n_s, embedding_dim):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    vocab_inp_size -- size of the python dictionary "vocab_inp_size"
    vocab_tar_size -- size of the python dictionary "vocab_tar_size"

    Returns:
    model -- Keras model instance
    """
    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X0 = Input(shape=(Tx, ),name='X')
    # (m,Tx)
    X = encoder_embedding(X0)
    # (m,Tx,embedding_dim)
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    Y = Input(shape=(Ty,),name='Y')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    decoder_X0 = Input(shape=(1,))
    decoder_X = decoder_X0
    #shape=(m, 1)
    #shape is not (m, Ty(10),vocab_tar_size) because we manually iterate Ty timesteps
    
    #Define encoder as Bi-LSTM
    encoder_output, forward_h, forward_c, backward_h, backward_c = encoder_layer(X)
    encoder_hidden = Concatenate(axis=-1)([forward_h, backward_h])
    encoder_cell = Concatenate(axis=-1)([forward_c, backward_c])
    
    s = encoder_hidden
    c = encoder_cell
    for t in range(Ty):
        y = Lambda(lambda x: x[:,t])(Y)
        y = reshape_layer(y)
        # one step of the attention mechanism to get back the context vector at step t 
        context = one_step_attention(encoder_output, encoder_hidden)
        # context--(m, 1, n_s)
        decoder_X = decoder_embedding(y)
        # (m,1) - (m,1,embedding_dim)
        decoder_inputs = enc_concat([decoder_X, context])
        # (m, 1, n_s+embedding_dim)
        # initial_state = [hidden state, cell state]
        decoder_X, s, c = decoder_cell(decoder_inputs, initial_state = [s, c])
        # decoder_X.shape--(m,n_s)
        # y and s are the same because of return_sequence=false by default
        # Apply Dense layer to the hidden state output of the decoder LSTM
        decoder_X = output_layer(decoder_X)
        #(m,vocab_tar_size)
        out = decoder_X
        outputs.append(out)
    
    # Create model instance taking three inputs and returning the list of outputs.
    model = Model([X0, s0, c0, decoder_X0, Y], outputs)    
    return model

# define model
model = model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, n_s, embedding_dim)
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='categorical_crossentropy')

# summarize defined model
print(model.summary())
#plot_model(model, to_file='model.png', show_shapes=True)
# fit model
#teacher forcing: np.append(np.zeros((trainY.shape[0],1)),trainY[:,:-1],1)
train_inputs = [trainX,np.zeros((trainX.shape[0],n_s)),np.zeros((trainX.shape[0],n_s)), np.zeros((trainX.shape[0], 1)),np.append(np.zeros((trainY.shape[0],1)),trainY[:,:-1],1)]
test_inputs = [testX,np.zeros((testX.shape[0],n_s)),np.zeros((testX.shape[0],n_s)), np.zeros((testX.shape[0], 1)),np.append(np.zeros((testY.shape[0],1)),testY[:,:-1],1)]
#swap first and second axes because optimizer expect shape (Ty,m)
train_outputs = list(trainY_oh.swapaxes(0,1))
test_outputs = list(testY_oh.swapaxes(0,1))
#filename = 'model.h5'
#checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(train_inputs, train_outputs, epochs=20, batch_size=64, validation_data=(test_inputs, test_outputs), callbacks=[early_stopping])

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
inf_model.save_weights('inf_model_wts.h5')
