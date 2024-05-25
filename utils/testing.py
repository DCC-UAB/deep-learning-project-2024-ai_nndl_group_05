import config
import wandb
import pandas as pd
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


def test(input_lang, output_lang, data_loader, type='test'):

    # Load Encoder and Decoder model
    encoder, decoder = loadEncoderDecoderModel(input_lang, output_lang)

    # Test
    encoder.eval()
    decoder.eval()

    criterion = {'NLLLoss': nn.NLLLoss, 'CrossEntropyLoss': nn.CrossEntropyLoss}[config.criterion]()
    
    translated_sentences = []
    total_loss = []
    total_acc = []

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):

            input_tensor, target_tensor = data
            input_tensor.to(device), target_tensor.to(device)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            acc = compute_accuracy(decoder_outputs, target_tensor)

            total_loss.append(loss.item())
            total_acc.append(acc)

            for input, output, target in zip(input_tensor, decoder_outputs, target_tensor):
                input_words, decoded_words, target_words = translate(input_lang, output_lang, 
                                                                    input, output, target)
                translated_sentences.append((input_words, decoded_words, target_words))

            if type == 'test':
                if batch_idx % config.batch_size == 0:
                    print(f'    Step [{batch_idx+1}/{len(data_loader)}], ' 
                        f' Loss: {loss.item():.4f}, '
                        f' Accuracy: {acc:.4f}')

    avg_loss = sum(total_loss) / len(data_loader)
    avg_acc = sum(total_acc) / len(data_loader)     
    
    # Print final metrics
    print(f'Average loss of {type} data: {avg_loss}, '
          f'Average accuracy of {type} data: {avg_acc}')
    
    # Store loss and accuracy evolution
    if type == 'test':
        wandb.log({'test/loss': avg_loss, 
                'test/accuracy': avg_acc})

    # Store translated sentences in csv
    df = pd.DataFrame(translated_sentences, columns=['Input', 'Output', 'Target'])
    
    if type == 'train':
        path = config.results_path_train
    elif type == 'val':
        path = config.results_path_val
    elif type == 'test':
        path = config.results_path_test

    df.to_csv(path, index=False)

    print("\nModel tested succesfully.")



# EVALUATE FUNCTIONS
"""def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn

def evaluateRandomly(encoder, decoder, n=10):
    encoder.eval()
    decoder.eval()
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')"""


    

