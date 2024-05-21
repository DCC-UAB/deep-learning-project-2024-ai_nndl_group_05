from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #dssable info messages
import config

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

import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, dropout=dropout, batch_first=True)
        
    def forward(self, inputs):
        outputs, (hidden, cell) = self.lstm(inputs)
        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, output_dim, hidden_dim, dropout):
        super(DecoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(output_dim, hidden_dim, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, inputs, hidden, cell):
        outputs, (hidden, cell) = self.lstm(inputs, (hidden, cell))
        predictions = self.softmax(self.fc(outputs))
        return predictions, hidden, cell

class EncoderGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EncoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
    def forward(self, inputs):
        outputs, hidden = self.gru(inputs)
        return hidden

class DecoderGRU(nn.Module):
    def __init__(self, output_dim, hidden_dim):
        super(DecoderGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(output_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, inputs, hidden):
        outputs, hidden = self.gru(inputs, hidden)
        predictions = self.softmax(self.fc(outputs))
        return predictions, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, cell_type):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.cell_type = cell_type
        
    def forward(self, encoder_input, decoder_input):
        if self.cell_type == 'LSTM':
            hidden, cell = self.encoder(encoder_input)
            decoder_output, _, _ = self.decoder(decoder_input, hidden, cell)
        elif self.cell_type == 'GRU':
            hidden = self.encoder(encoder_input)
            decoder_output, _ = self.decoder(decoder_input, hidden)
        
        return decoder_output

def modelTranslation(num_encoder_tokens, num_decoder_tokens):
    cell_type = config.cell_type
    latent_dim = config.latent_dim
    dropout = config.dropouts
    
    if cell_type == 'LSTM':
        encoder = EncoderLSTM(num_encoder_tokens, latent_dim, dropout)
        decoder = DecoderLSTM(num_decoder_tokens, latent_dim, dropout)
    elif cell_type == 'GRU':
        encoder = EncoderGRU(num_encoder_tokens, latent_dim)
        decoder = DecoderGRU(num_decoder_tokens, latent_dim)
    else:
        raise ValueError("Unsupported cell type: {}".format(cell_type))
    
    model = Seq2Seq(encoder, decoder, cell_type)
    
    return model


"""
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
"""

def trainSeq2Seq(model, train_loader, val_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, config.opt)(model.parameters(), lr=config.learning_rate)
    
    # TensorBoard writer
    log_path = "./models/log"
    writer = SummaryWriter(log_dir=log_path)
    
    # Training loop
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0
        for batch_idx, (encoder_inputs, decoder_inputs, targets) in enumerate(train_loader):

            encoder_inputs, decoder_inputs, targets = encoder_inputs.to(device), decoder_inputs.to(device), targets.to(device)
            print(len(encoder_inputs),len(decoder_inputs),len(targets))

            optimizer.zero_grad()
            outputs = model(encoder_inputs, decoder_inputs)
            print(len(outputs))

            #loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            outputs = outputs.view(config.batch_size, -1)
            targets = targets.view(config.batch_size,-1)
            print(len(outputs),len(targets))
            loss = criterion(outputs,targets)
            print(loss)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(epoch_loss)

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{config.epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        wandb.log({'train/loss': avg_epoch_loss})
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0
            for encoder_inputs, decoder_inputs, targets in val_loader:
                encoder_inputs, decoder_inputs, targets = encoder_inputs.to(device), decoder_inputs.to(device), targets.to(device)
                outputs = model(encoder_inputs, decoder_inputs)
                
                loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, -1)
                total += targets.size(0) * targets.size(1)  # assuming targets are of shape (batch_size, sequence_length)
                correct += (predicted == targets).sum().item()
                
            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            writer.add_scalar('Accuracy/validation', accuracy, epoch)
            wandb.log({'validation/loss': avg_val_loss, 'validation/accuracy': accuracy})
        
        model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for encoder_inputs, decoder_inputs, targets in val_loader:
            encoder_inputs, decoder_inputs, targets = encoder_inputs.to(device), decoder_inputs.to(device), targets.to(device)
            outputs = model(encoder_inputs, decoder_inputs)
            
            loss = criterion(outputs.view(-1, outputs.shape[-1]), targets.view(-1))
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, -1)
            total += targets.size(0) * targets.size(1)
            correct += (predicted == targets).sum().item()
            
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        wandb.log({'evaluate/loss': avg_val_loss, 'evaluate/accuracy': accuracy})

    writer.close()

"""
def trainSeq2Seq(model,encoder_dataset, decoder_input_dataset, decoder_target_dataset):
    # We load tensorboad
    # We train the model
    log_path="./models/log"
        
    tbCallBack = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=True, write_images=True)
    
    # Run training

    model.compile(optimizer=wandb.config.optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
    #categorical_crossentropy:  loss between the true classes and predicted classes. The labels are given in an one_hot format.

    train_dataset = tf.data.Dataset.zip((encoder_dataset, decoder_input_dataset))
    train_dataset = tf.data.Dataset.zip((train_dataset,  decoder_target_dataset))
    train_dataset = train_dataset.batch(wandb.config.batch_size)

    validation_dataset = train_dataset.take(int(config.validation_split * len(train_dataset)))
    train_dataset = train_dataset.skip(int(config.validation_split * len(train_dataset)))

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
"""

"""
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
        encoder_model.save(config.encoder_path)
        decoder_model.save(config.decoder_path)
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
        encoder_model.save(config.encoder_path)
        decoder_model.save(config.decoder_path)
        return encoder_model,decoder_model,reverse_target_char_index
"""    
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

def train(input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length, train_loader,val_loader):

    # we build the model
    model = modelTranslation(num_encoder_tokens,num_decoder_tokens)
    print("Model built successfully.\n")

    # we train it
    trainSeq2Seq(model, train_loader, val_loader)
    plot_model(model, to_file='./models/model.png', show_shapes=True, show_layer_names=True)
    print("\nModel trained successfully.\n")
    """
    # we build the final model for the inference (slightly different) and we save it
    encoder_model,decoder_model,reverse_target_char_index=generateInferenceModel(encoder_inputs, encoder_states,input_token_index,target_token_index,decoder_lstm,decoder_inputs,decoder_dense)
    plot_model(encoder_model, to_file='./models/model_encoder.png', show_shapes=True, show_layer_names=True)
    plot_model(decoder_model, to_file='./models/model_decoder.png', show_shapes=True, show_layer_names=True)
    print("\nFinal model built successfully.\n")

    # we save the object to convert the sequence to encoding  and encoding to sequence
    # our model is made for being used with different langages that do not have the same number of letters and the same alphabet
    saveChar2encoding("./models/char2encoding.pkl",input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index)
    print("Final model saved successfully.\n")"""