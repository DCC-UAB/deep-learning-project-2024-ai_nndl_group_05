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

def test(sentence):
    max_length  = config.max_length
    with torch.no_grad():
    input_tensor = tensorFromSentence(input_lang, input_sentence)
    input_length = input_tensor.size(0)

    encoder_outputs, encoder_hidden = encoder(input_tensor)

    decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS token as initial input
    decoder_hidden = encoder_hidden  # Initialize decoder hidden state with encoder's final hidden state

    decoded_words = []
    for di in range(max_length):
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        if topi.item() == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[topi.item()])

        decoder_input = topi.squeeze().detach()

    return ' '.join(decoded_words)


if __name__ == "__main__":
    #test(input("Enter a sentence to translate: "))
    input_lang, output_lang, train_loader, val_loader, test_loader = get_dataloader()
    sentence = "What is going on?"
    encoder,decoder = loadEncoderDecoderModel()
    output_translation = test(sentence, encoder, decoder, input_lang, output_lang)
    print('-')
    print("Input:", sentence)
    print("Translation:", output_translation)

