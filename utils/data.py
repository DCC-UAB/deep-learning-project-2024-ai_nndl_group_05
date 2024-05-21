from __future__ import print_function
import numpy as np
import tensorflow as tf
import config
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def extractChar(data_path, exchangeLanguage=False):
    # We extract the data (Sentence1 \t Sentence 2) from the anki text file
    input_texts = [] 
    target_texts = []
    input_characters = set()
    target_characters = set()
    #lines = open(data_path).read().split('\n')
    with open(data_path, encoding='utf-8') as file:
        lines = file.read().split('\n')

    if (exchangeLanguage==False):

        for line in lines[: min(config.num_samples, len(lines) - 1)]: 
            input_text, target_text, _ = line.split('\t')
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

    else:
        for line in lines[: min(config.num_samples, len(lines) - 1)]:
            target_text , input_text, _ = line.split('\t')
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))

    return input_characters,target_characters,input_texts,target_texts
    
    
def encodingChar(input_characters,target_characters,input_texts,target_texts):
    # We encode the dataset in a format that can be used by our Seq2Seq model (hot encoding).
    # Important: this project can be used for different language that do not have the same number of letter in their alphabet.
    # Important2: the decoder_target_data is ahead of decoder_input_data by one timestep (decoder = LSTM cell).
    # 1. We get the number of letter in language 1 and 2 (num_encoder_tokens/num_decoder_tokens)
    # 2. We create a dictonary for both language
    # 3. We store their encoding and return them and their respective dictonary
    
    num_encoder_tokens = len(input_characters) #numero de lletres diferents llengua entrada
    num_decoder_tokens = len(target_characters) #numero de lletres diferents llengua sortida
    max_encoder_seq_length = max([len(txt) for txt in input_texts]) #max len d'una linia entrada
    max_decoder_seq_length = max([len(txt) for txt in target_texts]) #max len d'una linia sortida

    print("#--------------data info 1---------------#")
    print('Number of num_encoder_tokens:', num_encoder_tokens)
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    print("#----------------------------------------#\n")
    
    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)]) # {"a": 0, "b": 1, "?": 2}
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length



def create_data_loader1(encoder_input_data, decoder_input_data, decoder_target_data):
    # Create TensorFlow datasets from the encoded data arrays
    encoder_dataset = tf.data.Dataset.from_tensor_slices(encoder_input_data)
    decoder_input_dataset = tf.data.Dataset.from_tensor_slices(decoder_input_data)
    decoder_target_dataset = tf.data.Dataset.from_tensor_slices(decoder_target_data)

    print("#--------data info 2 (tensorflow)-------#")
    print("encoder_dataset (tf): ",len(encoder_dataset), type(encoder_dataset))
    print("decoder_input_dataset (tf): ",len(decoder_input_dataset),type(decoder_input_dataset))
    print("decoder_target_dataset (tf): ",len(decoder_target_dataset),type(decoder_target_dataset))
    print("#---------------------------------------#")
        
    return encoder_dataset, decoder_input_dataset, decoder_target_dataset 

def create_data_loader2(encoder_input_data, decoder_input_data, decoder_target_data, batch_size=32, shuffle=True):
    # Convert numpy arrays to PyTorch tensors if they are numpy arrays
    if isinstance(encoder_input_data, np.ndarray):
        encoder_input_data = torch.tensor(encoder_input_data)
    if isinstance(decoder_input_data, np.ndarray):
        decoder_input_data = torch.tensor(decoder_input_data)
    if isinstance(decoder_target_data, np.ndarray):
        decoder_target_data = torch.tensor(decoder_target_data)
    
    # Create datasets and data loaders
    train_dataset = TensorDataset(encoder_input_data, decoder_input_data, decoder_target_data)
    val_size = int(config.validation_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    print("#-----------data info 3 (pytorch)----------#")
    print("train_dataset",len(train_dataset),type(train_dataset))
    print("val_size",val_size)
    print("train_size",train_size)
    print("train_dataset",len(train_dataset),type(train_dataset))
    print("val_dataset",len(val_dataset),type(val_dataset))
    print("train_loader:",len(train_loader),type(train_loader))
    print("val_loader:",len(val_loader),type(val_loader))
    print("#----------------------------------------#")

    return train_loader, val_loader


#--------------------MAIN----------------------#

def prepareData(data_path):

    input_characters,target_characters,input_texts,target_texts=extractChar(data_path)

    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,num_encoder_tokens,num_decoder_tokens,num_decoder_tokens,max_encoder_seq_length =encodingChar(input_characters,target_characters,input_texts,target_texts)
        
    encoder_dataset, decoder_input_dataset, decoder_target_dataset  = create_data_loader1(encoder_input_data, decoder_input_data, decoder_target_data)
    train_loader, val_loader  = create_data_loader2(encoder_input_data, decoder_input_data, decoder_target_data)
    
    
    print("#--------------data info 4---------------#")
    print("encoder_input_data:",len(encoder_input_data),type(encoder_input_data))
    print("decoder_input_data:",len(decoder_input_data),type(decoder_input_data))
    print("decoder_target_data:",len(decoder_target_data),type(decoder_target_data))

    print("input_token_index:",len(input_token_index),type(input_token_index))
    print("target_token_index:",len(target_token_index),type(target_token_index))

    print("input_texts:",len(input_texts),type(input_texts))
    print("target_texts:",len(target_texts),type(target_texts))

    print("num_encoder_tokens:",num_encoder_tokens,type(num_encoder_tokens))
    print("num_decoder_tokens:",num_decoder_tokens,type(num_decoder_tokens))
    print("max_encoder_seq_length:",max_encoder_seq_length,type(max_encoder_seq_length))

    print("#----------------------------------------#\n")
    return encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length, train_loader, val_loader
