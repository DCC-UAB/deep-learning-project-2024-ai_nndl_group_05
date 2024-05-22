from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #dssable info messages
import config

from keras.utils.vis_utils import plot_model
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
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

"""
class Seq2SeqEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, n_layers, cell_type='LSTM'):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True)
        else:
            raise ValueError("Invalid cell type. Use 'LSTM' or 'GRU'.")
    
    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Seq2SeqDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, n_layers, cell_type='LSTM'):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, trg, hidden):
        embedded = self.embedding(trg)
        outputs, hidden = self.rnn(embedded, hidden)
        predictions = self.fc(outputs)
        return predictions, hidden
"""
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

def modelTranslation():
    cell_type = config.cell_type
    input_dim = config.input_dim
    latent_dim = config.latent_dim
    output_dim = config.output_dim
    dropout = config.dropouts
    
    #encoder = Seq2SeqEncoder(input_dim, latent_dim, hidden_dim, config.ltsm_layers, config.cell_type)
    #decoder = Seq2SeqDecoder(output_dim, latent_dim, hidden_dim, config.ltsm_layers, config.cell_type)
    
    if cell_type == 'LSTM':
        encoder = EncoderLSTM(input_dim, latent_dim, dropout)
        decoder = DecoderLSTM(output_dim, latent_dim, dropout)
    elif cell_type == 'GRU':
        encoder = EncoderGRU(input_dim, latent_dim)
        decoder = DecoderGRU(output_dim, latent_dim)
    else:
        raise ValueError("Unsupported cell type: {}".format(cell_type))
    

    model = Seq2Seq(encoder, decoder, cell_type)
    
    return model

def compute_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / y[non_pad_elements].shape[0]

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
        epoch_acc = 0
        for batch_idx, (encoder_inputs, decoder_inputs, targets) in enumerate(train_loader):

            encoder_inputs, decoder_inputs, targets = encoder_inputs.to(device), decoder_inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            outputs = model(encoder_inputs, decoder_inputs)
   
            loss = criterion(outputs,targets)
            acc = compute_accuracy(outputs, targets)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{config.epochs}], Step [{batch_idx+1}/{len(train_loader)}],' 
                      f'Loss: {loss.item():.4f}, Accuracy: {acc.item():.4f}')
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_acc = epoch_acc / len(train_loader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', avg_epoch_acc, epoch)
        wandb.log({'train/loss': avg_epoch_loss, 'train/accuracy': avg_epoch_acc})
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            #val_acc = 0
            correct = 0
            total = 0
            for encoder_inputs, decoder_inputs, targets in val_loader:
                encoder_inputs, decoder_inputs, targets = encoder_inputs.to(device), decoder_inputs.to(device), targets.to(device)
                
                outputs = model(encoder_inputs, decoder_inputs)
                
                loss = criterion(outputs,targets)

                val_loss += loss.item()
                #val_acc += acc.item()
                
                #_, predicted = torch.max(outputs.data, -1)
                _, predicted = torch.max(outputs.data)
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
                
            loss = criterion(outputs,targets)

            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, -1)
            total += targets.size(0) * targets.size(1)
            correct += (predicted == targets).sum().item()
            
        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        wandb.log({'evaluate/loss': avg_val_loss, 'evaluate/accuracy': accuracy})

    writer.close()

    plot_model(model, to_file=config.png_model_path, show_shapes=True, show_layer_names=True)
    # Save model with torch or with onnx

def generateInferenceModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define the model parameters
    input_dim = config.input_dim
    output_dim = config.output_dim
    latent_dim = config.latent_dim #embed_dim = wandb.config.embed_dim
    #hidden_dim = wandb.config.latent_dim
    #n_layers = wandb.config.n_layers
    cell_type = config.cell_type
    dropout = config.dropouts
    
    # Initialize encoder and decoder
    #encoder = Seq2SeqEncoder(input_dim, latent_dim, hidden_dim, n_layers, cell_type).to(device)
    #decoder = Seq2SeqDecoder(output_dim, latent_dim, hidden_dim, n_layers, cell_type).to(device)
    if cell_type == 'LSTM':
        encoder = EncoderLSTM(input_dim, latent_dim, dropout).to(device)
        decoder = DecoderLSTM(output_dim, latent_dim, dropout).to(device)
    elif cell_type == 'GRU':
        encoder = EncoderGRU(input_dim, latent_dim).to(device)
        decoder = DecoderGRU(output_dim, latent_dim).to(device)
    else:
        raise ValueError("Invalid cell type. Use 'LSTM' or 'GRU'.")

    # Save models
    torch.save(encoder.state_dict(), config.encoder_path)
    torch.save(decoder.state_dict(), config.decoder_path)

    plot_model(encoder, to_file=config.png_encoder_path, show_shapes=True, show_layer_names=True)
    plot_model(decoder, to_file=config.png_decoder_path, show_shapes=True, show_layer_names=True)
    



#--------------------MAIN----------------------#

def train(train_loader,val_loader):

    # we build the model
    model = modelTranslation()
    print("Model built successfully.\n")

    # we train it
    trainSeq2Seq(model, train_loader, val_loader)
    print("\nModel trained successfully.\n")
    
    # we build the final model for the inference (slightly different) and we save it
    generateInferenceModel()
    print("\nInference model built successfully.\n")
