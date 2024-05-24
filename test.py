import numpy as np
import config
#import wandb
import pickle
import torch.nn as nn
import torch
from utils.training import EncoderRNN, DecoderRNN
from utils.data import get_dataloader, normalizeString

#first we will load the model, both the encoder, and the decoder
def loadEncoderDecoderModel():
    encoder = EncoderRNN(input_lang.n_words, config.latent_dim)
    decoder =  DecoderRNN(config.latent_dim, output_lang.n_words)
    encoder.load_state_dict(torch.load(config.encoder_path))
    decoder.load_state_dict(torch.load(config.decoder_path))
    return encoder, decoder

def test(sentence,encoder, decoder, input_lang, output_lang, max_length=config.max_length):
    max_length  = config.max_length
    with torch.no_grad():
        input_tensor = indexesFromSentences(input_lang, sentence)
        input_length = input_tensor.size(0)
        encoder.to('cuda')
        decoder.to('cuda')

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        print(encoder_hidden)
        decoder_input = torch.tensor([0], device='cuda')  #SOS token as initial input (0)
        decoder_hidden = encoder_hidden.to('cuda') #initialize decoder hidden state with encoder's final hidden state
        decoded_words = []
        for di in range(max_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            if topi.item() == 1: #EOS idx
                decoded_words.append('<EOS>') #represent end of sentence
                break #if eos, stop
            else:
                decoded_words.append(output_lang.index2word[topi.item()]) #else append word to the translation

            decoder_input = topi.squeeze().detach() #pass it as the next input for the decoder

        return ' '.join(decoded_words) #return translation

def indexesFromSentences(lang, sentence):
    return torch.tensor([lang.word2index[word] for word in sentence.split(' ')], dtype=torch.long, device='cuda').view(-1, 1)

if __name__ == "__main__":
    #test(input("Enter a sentence to translate: "))
    input_lang, output_lang, train_loader, val_loader, test_loader = get_dataloader()
    sentence = "who am i"
    sentence = normalizeString(sentence)
    encoder,decoder = loadEncoderDecoderModel()
    output_translation = test(sentence, encoder, decoder, input_lang, output_lang)
    print('-')
    print("Input:", sentence)
    print("Translation:", output_translation)

