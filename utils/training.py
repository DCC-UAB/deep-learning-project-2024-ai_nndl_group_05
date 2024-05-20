from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #dssable info messages

from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU
from keras.callbacks import TensorBoard
#import _pickle as pickle
import pickle
import wandb
#from wandb.keras import WandbCallback
from wandb.integration.keras import WandbCallback
import tensorflow as tf
from keras.utils.vis_utils import plot_model
#from tensorflow.keras.utils import plot_model

#-----------------GLOBAL VARIABLES-------------------#

batch_size = 128  # Batch size for training.
epochs = 3  # Number of epochs to train for.
latent_dim = 256 #1024 # Latent dimensionality of the encoding space.
num_samples =  90000 # Number of samples to train on.
validation_split = 0.2
learning_rate = 0.0001

LOG_PATH='./log'
# Path to the data txt file on disk.
# './cat-eng/cat.txt' el dataset en catala nomes te 1336 linies
data_path = './spa-eng/spa.txt' #139705 lines
encoder_path='/models/encoder_modelTranslation.h5'
decoder_path='/models/decoder_modelTranslation.h5'


name = "Execution"
opt = 'rmsprop' #'adam'

#--------------------FUNCTIONS----------------------#

def modelTranslation(num_encoder_tokens,num_decoder_tokens):
# We crete the model 1 encoder(lstm)/1 encoder(gru) + 1 decode (LSTM)/1 decode (gru) + 1 Dense layer + softmax

    if wandb.config.cell_type == 'LSTM':
    
        encoder_inputs = Input(shape=(None, num_encoder_tokens)) 
        # input tensor to the encoder. It has a shape of (None, num_encoder_tokens), where None represents the variable-length 
        # sequence and num_encoder_tokens is the number of tokens in the input lenguage.
        encoder = LSTM(wandb.config.latent_dim, return_state=True, dropout=wandb.config.dropouts)
        # latent_dim: Latent dimensionality of the encoding space.
        # encoder LSTM layer is created with latent_dim units
        encoder_outputs, state_h, state_c = encoder(encoder_inputs) 
        # encoder_outputs: output sequence from the encoder LSTM layer
        # state_h and state_c: final hidden state and cell state of the encoder.
        encoder_states = [state_h, state_c]
        # will be used as the initial state for the decoder
        
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_lstm = LSTM(wandb.config.latent_dim, return_sequences=True, return_state=True, dropout=wandb.config.dropouts)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                                initial_state=encoder_states)

        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        

        return model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense

    
    elif wandb.config.cell_type =='GRU':
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = GRU(wandb.config.latent_dim, return_state=True)
        encoder_outputs, state_h = encoder(encoder_inputs)
        encoder_states = state_h

        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        decoder_gru = GRU(wandb.config.latent_dim, return_sequences=True)
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
        return model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_gru,decoder_dense

def trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data, encoder_dataset, decoder_input_dataset, decoder_target_dataset):
# We load tensorboad
# We train the model
    LOG_PATH="./output/log"
        
    tbCallBack = TensorBoard(log_dir=LOG_PATH, histogram_freq=0, write_graph=True, write_images=True)
    
    # Run training

    model.compile(optimizer=wandb.config.optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    #categorical_crossentropy:  loss between the true classes and predicted classes. The labels are given in an one_hot format.

    train_dataset = tf.data.Dataset.zip((encoder_dataset, decoder_input_dataset))
    train_dataset = tf.data.Dataset.zip((train_dataset,  decoder_target_dataset))
    train_dataset = train_dataset.batch(wandb.config.batch_size)

    validation_dataset = train_dataset.take(int(validation_split * len(train_dataset)))
    train_dataset = train_dataset.skip(int(validation_split * len(train_dataset)))

    model.fit(train_dataset, batch_size=wandb.config.batch_size, epochs=wandb.config.epochs, validation_data=validation_dataset, callbacks=[WandbCallback()])
    
    # model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
    #             batch_size=batch_size,
    #             epochs=epochs,
    #             validation_split=0.01,
    #             callbacks = [tbCallBack])
    
    
    # Evaluate    
    #loss, accuracy = model.evaluate(validation_dataset, callbacks=[WandbCallback()])
    loss, acc = model.evaluate(validation_dataset)
    wandb.log({'evaluate/accuracy': acc})


def generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense):
# Once the model is trained, we connect the encoder/decoder and we create a new model
# Finally we save everything
    if wandb.config.cell_type == 'LSTM':
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(wandb.config.latent_dim,))
        decoder_state_input_c = Input(shape=(wandb.config.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)
        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_input_char_index = dict(
            (i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())
        encoder_model.save(encoder_path)
        decoder_model.save(decoder_path)
        return encoder_model,decoder_model,reverse_target_char_index
    
    elif wandb.config.cell_type == 'GRU':
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(wandb.config.latent_dim,))
        #decoder_state_input_c = Input(shape=(wandb.config.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h]
        #decoder_outputs, state_h = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_outputs, state_h = GRU(wandb.config.latent_dim, return_sequences=True, return_state=True)(decoder_inputs, initial_state=decoder_states_inputs[0])
        decoder_states = [state_h]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_input_char_index = dict(
            (i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())
        encoder_model.save(encoder_path)
        decoder_model.save(decoder_path)
        return encoder_model,decoder_model,reverse_target_char_index
    
def saveChar2encoding(filename,input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index):
    f = open(filename, "wb")
    pickle.dump(input_token_index, f)
    pickle.dump(max_encoder_seq_length, f)
    pickle.dump(num_encoder_tokens, f)
    pickle.dump(reverse_target_char_index, f)
    
    pickle.dump(num_decoder_tokens, f)
    
    pickle.dump(target_token_index, f)
    f.close()



#--------------------MAIN----------------------#

def train(encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length, encoder_dataset, decoder_input_dataset, decoder_target_dataset):

    # we build the model
    model,decoder_outputs,encoder_inputs,encoder_states,decoder_inputs,decoder_lstm,decoder_dense=modelTranslation(num_encoder_tokens,num_decoder_tokens)
    print("Model built successfully.\n")

    # we train it
    trainSeq2Seq(model,encoder_input_data, decoder_input_data,decoder_target_data, encoder_dataset, decoder_input_dataset, decoder_target_dataset)
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    print("\nModel trained successfully.\n")

    # we build the final model for the inference (slightly different) and we save it
    encoder_model,decoder_model,reverse_target_char_index=generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense)
    #plot_model(encoder_model, to_file='model_encoder.png', show_shapes=True, show_layer_names=True)
    #plot_model(decoder_model, to_file='model_decoder.png', show_shapes=True, show_layer_names=True)
    print("\nFinal model built successfully.\n")

    # we save the object to convert the sequence to encoding  and encoding to sequence
    # our model is made for being used with different langages that do not have the same number of letters and the same alphabet
    saveChar2encoding("./models/char2encoding.pkl",input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index)
    print("Final model saved successfully.\n")