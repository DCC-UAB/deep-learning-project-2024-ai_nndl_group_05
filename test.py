import numpy as np
import config
#import wandb
import pickle
import torch.nn as nn
import torch
from utils.training import EncoderRNN, DecoderRNN
from utils.data import get_dataloader

#first we will load the model, both the encoder, and the decoder
def loadEncoderDecoderModel():
    encoder = EncoderRNN(input_lang.n_words, config.latent_dim)
    decoder =  DecoderRNN(config.latent_dim, output_lang.n_words)
    encoder.load_state_dict(torch.load(config.encoder_path))
    decoder.load_state_dict(torch.load(config.decoder_path))
    return encoder, decoder




if __name__ == "__main__":
    #test(input("Enter a sentence to translate: "))
    input_lang, output_lang, train_loader, val_loader, test_loader = get_dataloader()
    sentence = "What is going on?"
    encoder,decoder = loadEncoderDecoderModel()
    print(encoder)
    print("\n", decoder)
    #test(sentence)
