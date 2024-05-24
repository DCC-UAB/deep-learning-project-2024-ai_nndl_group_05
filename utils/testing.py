import config
import wandb
import torch
import torch.nn as nn
from utils.training import EncoderRNN, DecoderRNN, translate, compute_accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadEncoderDecoderModel(input_lang, output_lang):
    encoder = EncoderRNN(input_lang.n_words, config.latent_dim)
    decoder =  DecoderRNN(config.latent_dim, output_lang.n_words)
    encoder.load_state_dict(torch.load(config.encoder_path))
    decoder.load_state_dict(torch.load(config.decoder_path))
    return encoder, decoder


def test(input_lang, output_lang, test_loader):

    # Load Encoder and Decoder model
    encoder, decoder = loadEncoderDecoderModel(input_lang, output_lang)

    # Test
    criterion = {'NLLLoss': nn.NLLLoss, 'CrossEntropyLoss': nn.CrossEntropyLoss}[config.criterion]()
    
    translated_sentences = []
    test_loss = []
    test_acc = []

    with torch.no_grad():
        for batch_idx, data in test_loader:

            input_tensor, target_tensor = data
            input_tensor.to(device), target_tensor.to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )

            test_loss.append(loss.item())
            test_acc.append(compute_accuracy(decoder_outputs, target_tensor))

            for input, output, target in zip(input_tensor, decoder_outputs, target_tensor):
                input_words, decoded_words, target_words = translate(input_lang, output_lang, 
                                                                    input, output, target)
                translated_sentences.append((input_words, decoded_words, target_words))

            if batch_idx % config.batch_size == 0:
                print(f'    Step [{batch_idx+1}/{len(test_loader)}], ' 
                      f' Loss: {loss.item():.4f}, '
                      f' Accuracy: ')
    
    return translated_sentences

