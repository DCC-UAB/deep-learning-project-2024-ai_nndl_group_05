import config
import wandb
import pandas as pd
import torch
import torch.nn as nn
from utils.training import EncoderRNN, DecoderRNN
from utils.training import compute_accuracy, evaluate_wer, translate, evaluate_cer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadEncoderDecoderModel(input_lang, output_lang):
    if config.model == "words":
        encoder = EncoderRNN(input_lang.n_words, config.latent_dim).to(device)
        decoder =  DecoderRNN(config.latent_dim, output_lang.n_words).to(device)
    elif config.model == "chars":
        encoder = EncoderRNN(input_lang.n_chars, config.latent_dim).to(device)
        decoder = DecoderRNN(config.latent_dim, output_lang.n_chars).to(device)
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
    total_wer = []
    total_cer = []

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
            acc = compute_accuracy(decoder_outputs, target_tensor, output_lang)

            total_loss.append(loss.item())
            total_acc.append(acc)

            if config.model == "words":
                # Compute WER and PER for the current batch
                wer = evaluate_wer(decoder_outputs, target_tensor, output_lang)
                total_wer.append(wer)
                
            elif config.model == "chars":
                # Compute WER and PER for the current batch
                cer = evaluate_cer(decoder_outputs, target_tensor, output_lang)
                total_cer.append(cer)
            
            for input, output, target in zip(input_tensor, decoder_outputs, target_tensor):
                input_words, decoded_words, target_words = translate(input_lang, output_lang, 
                                                                    input, output, target)
                translated_sentences.append((input_words, decoded_words, target_words))

            if type == 'test':
                if batch_idx % config.batch_size == 0:
                    if config.model == "words":
                        print(f'    Step [{batch_idx+1}/{len(data_loader)}], ' 
                            f' Loss: {loss.item():.4f}, '
                            f'Accuracy: {acc:.4f}, '
                            f'WER: {wer:.4f}')
                    elif config.model == "chars":
                        print(f'    Step [{batch_idx+1}/{len(data_loader)}], ' 
                            f' Loss: {loss.item():.4f}, '
                            f'Accuracy: {acc:.4f},'
                            f'CER: {cer:.4f}')

    avg_loss = sum(total_loss) / len(data_loader)
    avg_acc = sum(total_acc) / len(data_loader) 

    if config.model == "words": 
        avg_wer = sum(total_wer) / len(data_loader)
        
        # Print final metrics
        print(f'Average loss of {type} data: {avg_loss:.4f}, '
            f'Average accuracy of {type} data: {avg_acc:.4f}, '
            f'Average WER of {type} data: {avg_wer:.4f}')
        
        # Store loss and accuracy evolution
        if type == 'test':
            if config.do_wandb:
                wandb.log({'test/loss': avg_loss, 'test/accuracy': avg_acc, 'test/WER': avg_wer})
            
    elif config.model == "chars":
        avg_cer = sum(total_cer) / len(data_loader)

        
        print(f'Average loss of {type} data: {avg_loss:.4f}, '
            f'Average accuracy of {type} data: {avg_acc:.4f}, '
            f'Average CER of {type} data: {avg_cer:.4f}')
        # Store loss and accuracy evolution
        if type == 'test':
            if config.do_wandb:
                    wandb.log({'test/loss': avg_loss, 'test/accuracy': avg_acc, 'test/CER': avg_cer})
    

    # Store translated sentences in csv
    df = pd.DataFrame(translated_sentences, columns=['Input', 'Output', 'Target'])
    
    if type == 'train':
        path = config.results_path_train
    elif type == 'val':
        path = config.results_path_val
    elif type == 'test':
        path = config.results_path_test

    df.to_csv(path, index=False)
