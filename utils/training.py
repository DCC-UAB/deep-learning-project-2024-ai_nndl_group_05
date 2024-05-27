from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#from torchviz import make_dot

#from nltk.translate.bleu_score import corpus_bleu
import jiwer

import time
import math
import wandb

import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


# FUNCTIONS

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

# EVALUATIONS
def compute_accuracy(predictions, targets):
    _, predicted_ids = predictions.max(dim=-1)  # Get the index of the max log-probability
    
    correct = (predicted_ids == targets).float()  # Compare predictions with targets
    accuracy = correct.sum() / correct.numel()  # Calculate accuracy as percentage
    return accuracy.item()

def wer(reference, hypothesis):
    wer = jiwer.wer(reference, hypothesis)
    return wer

def per(reference, hypothesis):
    total_error = 0
    c = 0
    for ref, hyp in zip(reference, hypothesis):
        # Convert word lists to character lists
        ref_chars = ' '.join(ref)
        hyp_chars = ' '.join(hyp)
        # Calculate character-level WER, which is effectively PER
        error = jiwer.wer(ref_chars, hyp_chars)
        total_error += error
        c += 1
    return total_error / c

def evaluate_per(predictions, targets, output_lang):
    def tensor_to_words(tensor, lang):
        words = []
        for idx in tensor:
            words.append(lang.index2word[idx.item()])
        return words

    total_per = 0
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        predicted_ids = predictions[i].max(dim=-1)[1]  # Get the predicted word indices
        reference_sentence = tensor_to_words(targets[i], output_lang)
        hypothesis_sentence = tensor_to_words(predicted_ids, output_lang)
        
        per_value = per(reference_sentence, hypothesis_sentence)
        total_per += per_value
    
    average_per = total_per / batch_size
    return average_per

def evaluate_wer(predictions, targets, output_lang):
    def tensor_to_words(tensor, lang):
        """
        Convert tensor of word indices to a list of words.
        """
        words = []
        for idx in tensor:
            words.append(lang.index2word[idx.item()])
        return words

    total_wer = 0
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        predicted_ids = predictions[i].max(dim=-1)[1]  # Get the predicted word indices
        reference_sentence = tensor_to_words(targets[i], output_lang)
        hypothesis_sentence = tensor_to_words(predicted_ids, output_lang)
        
        wer_value = wer(reference_sentence, hypothesis_sentence)
        total_wer += wer_value
    
    average_wer = total_wer / batch_size
    return average_wer

def translate(input_lang, output_lang, 
              input_tensor, decoded_outputs, target_tensor):

    def get_words(lang, tensor):

        _, topi = tensor.topk(1)
        ids = topi.squeeze()

        words = []
        for idx in ids:
            if idx.item() == EOS_token:
                words.append('EOS')
                break
            words.append(lang.index2word[idx.item()])

        return words

    input_words = [input_lang.index2word[idx.item()] for idx in input_tensor]
    decoded_words = get_words(output_lang, decoded_outputs)
    target_words = [output_lang.index2word[idx.item()] for idx in target_tensor]

    return input_words, decoded_words, target_words


# ENCODER / DECODER
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=config.dropouts):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(config.max_length):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    

# TRAINING AND VALIDATION EPOCHS

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion,output_lang, n_epoch):

    total_loss = 0
    total_acc = 0
    total_wer = 0
    total_per = 0

    for batch_idx, data in enumerate(dataloader):
        input_tensor, target_tensor = data
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        acc = compute_accuracy(decoder_outputs, target_tensor)
        total_acc += acc
        wer = evaluate_wer(decoder_outputs, target_tensor, output_lang)
        total_wer += wer 
        per = evaluate_per(decoder_outputs, target_tensor, output_lang)
        total_per += per  

        if batch_idx % config.batch_size == 0:
            print(f'    Step [{batch_idx+1}/{len(dataloader)}], ' 
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {acc:.4f}, '
                  f'WER: {wer:.4f}, '
                  f'PER: {per:.4f}')

    """# If it is the last epoch, save the model visualization
    if n_epoch == config.epochs:
        y = encoder(input_tensor)
        make_dot(y, params=dict(list(encoder.named_parameters()))).render(config.png_encoder_path, format="png")
        y, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        make_dot(y, params=dict(list(decoder.named_parameters()))).render(config.png_decoder_path, format="png")"""

    average_loss = total_loss / len(dataloader)  # Calculate average loss over all batches
    average_acc = total_acc / len(dataloader)   # Calculate average accuracy over all batches
    average_wer = total_wer / len(dataloader)  # Calculate average WER over all batches
    average_per= total_per / len(dataloader)  # Calculate average PER over all batches
    return average_loss, average_acc, average_wer, average_per


def val_epoch(dataloader, encoder, decoder, criterion,
              input_lang, output_lang):
    
    total_loss = 0
    total_acc = 0
    total_wer = 0 
    total_per = 0

    for batch_idx, data in enumerate(dataloader):
        
        input_tensor, target_tensor = data
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        acc = compute_accuracy(decoder_outputs, target_tensor)

        total_loss += loss.item()
        total_acc += acc

        # Compute WER for the current batch
        wer = evaluate_wer(decoder_outputs, target_tensor, output_lang)
        total_wer += wer  # Sum the batch-wise average WER

        per = evaluate_per(decoder_outputs, target_tensor, output_lang)
        total_per += per  # Sum the batch-wise BLEU score

        if batch_idx % config.batch_size == 0:
            print(f'        Step [{batch_idx+1}/{len(dataloader)}], ' 
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {acc:.4f}, '
                  f'WER: {wer:.4f}, '
                  f'PER: {per:.4f}')

            # Get translation examples
            input_words, decoded_words, target_words = translate(input_lang, output_lang, 
                                                                 input_tensor[0], 
                                                                 decoder_outputs[0], 
                                                                 target_tensor[0])
            
            print(f'            {input_lang.name}: {input_words}')
            print(f'            {output_lang.name} translation: {decoded_words}')
            print(f'            {output_lang.name} ground truth: {target_words}')

    average_loss = total_loss / len(dataloader)
    average_acc = total_acc / len(dataloader)
    average_wer = total_wer / len(dataloader) 
    average_per = total_per/len(dataloader)
    return average_loss, average_acc, average_wer, average_per


# TRAINING LOOP

def trainSeq2Seq(train_loader, val_loader, encoder, decoder,
                 input_lang, output_lang):
    
    start = time.time()
    
    losses_train, acc_train, wer_train, per_train = [],[],[],[]
    losses_val, acc_val, wer_val, per_val = [],[],[],[]

    # Define optimizer and criterion
    encoder_optimizer = getattr(torch.optim, config.opt)(encoder.parameters(), lr=config.learning_rate)
    print("Encoder optimizer:",encoder_optimizer)

    decoder_optimizer = getattr(torch.optim, config.opt)(decoder.parameters(), lr=config.learning_rate)
    print("Decoder optimizer:",decoder_optimizer)
    
    criterion = {'NLLLoss': nn.NLLLoss, 'CrossEntropyLoss': nn.CrossEntropyLoss}[config.criterion]()
    print("Loss function: ", criterion)

    # Training
    encoder.train()
    decoder.train()

    for epoch in range(1, config.epochs + 1):

        print("\nEpoch:",epoch)

        #loss = train_epoch(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        loss, acc, wer, per = train_epoch(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang, n_epoch=epoch)

        losses_train.append(loss)
        acc_train.append(acc)
        wer_train.append(wer)
        per_train.append(per)


        print(f'    Time: {timeSince(start, epoch / config.epochs)}, '
              f'Epochs completed: {epoch / config.epochs * 100}%, '
              f'Epoch loss: {loss:.4f}, '
              f'Epoch accuracy: {acc:.4f}, '
              f'Epoch WER: {wer:.4f}, '
              f'Epoch PER: {per:.4f}')

        wandb.log({'epoch': epoch, 'train/loss': loss, 'train/accuracy': acc, 'train/WER': wer, 'train/PER': per})

        # Validation
        encoder.eval()
        decoder.eval()

        with torch.no_grad():

            print(f'\n   Validation: epoch {epoch}')
            
            #val_loss = val_epoch(val_loader, encoder, decoder, criterion, input_lang, output_lang)
            val_loss, val_acc, val_wer, val_per = val_epoch(val_loader, encoder, decoder, criterion, input_lang, output_lang)

            losses_val.append(val_loss)
            acc_val.append(val_acc)
            wer_val.append(val_wer)
            per_val.append(val_per)

            wandb.log({'epoch': epoch, 'validation/loss': val_loss, 'validation/accuracy': val_acc, 'validation/WER': val_wer, 'validation/PER': val_per})
        
        encoder.train()
        decoder.train()

    # Save the trained models
    torch.save(encoder.state_dict(), config.encoder_path)
    torch.save(decoder.state_dict(), config.decoder_path)



# MAIN
def train(input_lang, output_lang, train_loader, val_loader):
    # Create encoder and decoder
    encoder = EncoderRNN(input_lang.n_words, config.latent_dim).to(device)
    decoder = DecoderRNN(config.latent_dim, output_lang.n_words).to(device)
    print("Encoder and decoder created.\n")

    # Train the decoder and encoder
    trainSeq2Seq(train_loader, val_loader, encoder, decoder, input_lang, output_lang)
    print("\nModel trained successfully.")
