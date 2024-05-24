from __future__ import unicode_literals, print_function, division

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
#from torchviz import make_dot

import time
import math

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

def compute_accuracy(decoder_outputs, target_tensor):

    total_correct_words = 0

    for output, target in zip(decoder_outputs, target_tensor):
        
        # len(output) != len(target)

        _, topi = output.topk(1)
        pred = topi.squeeze()

        correct = 0

        # ... NOT FINISHED 

    return total_correct_words / (len(decoder_outputs) * config.max_length)

def translate(input_lang, output_lang, 
              input_tensor, decoded_outputs, target_tensor):

    def get_words(lang, tensor):

        _, topi = tensor.topk(1)
        ids = topi.squeeze()

        words = []
        for idx in ids:
            if idx.item() == EOS_token:
                words.append('<EOS>')
                break
            words.append(lang.index2word[idx.item()])

        return words

    input_words = [input_lang.index2word[idx.item()] for idx in input_tensor]
    decoded_words = get_words(output_lang, decoded_outputs)
    target_words = [output_lang.index2word[idx.item()] for idx in target_tensor]

    return input_words, decoded_words, target_words


# ENCODER / DECODER
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
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
          decoder_optimizer, criterion):

    total_loss = 0
    total_acc = 0

    for batch_idx, data in enumerate(dataloader):

        input_tensor, target_tensor = data
        input_tensor.to(device), target_tensor.to(device)

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
        #total_acc += compute_accuracy(decoder_outputs, target_tensor)

        if batch_idx % config.batch_size == 0:
                print(f'    Step [{batch_idx+1}/{len(dataloader)}], ' 
                      f' Loss: {loss.item():.4f}, '
                      f' Accuracy: ')

    return total_loss / len(dataloader)
    #return total_loss / len(dataloader), total_acc / len(dataloader)


def val_epoch(dataloader, encoder, decoder, criterion,
              input_lang, output_lang):
    
    total_loss = 0
    total_acc = 0

    for batch_idx, data in enumerate(dataloader):
        
        input_tensor, target_tensor = data
        input_tensor.to(device), target_tensor.to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )

        total_loss += loss.item()
        #total_acc += compute_accuracy(decoder_outputs, target_tensor)

        if batch_idx % config.batch_size == 0:
            print(f'        Step [{batch_idx+1}/{len(dataloader)}], ' 
                  f' Loss: {loss.item():.4f}, ')
                  #f' Accuracy: ')

            # Get translation examples
            input_words, decoded_words, target_words = translate(input_lang, output_lang, 
                                                                input_tensor[0], 
                                                                decoder_outputs[0], 
                                                                target_tensor[0])
            
            print(f'            {input_lang.name}: {input_words}')
            print(f'            {output_lang.name} translation: {decoded_words}')
            print(f'            {output_lang.name} ground truth: {target_words}')

    return total_loss / len(dataloader)
    #return total_loss / len(dataloader), total_acc / len(dataloader)


# TRAINING LOOP

def trainSeq2Seq(train_loader, val_loader, encoder, decoder,
                 input_lang, output_lang):
    
    start = time.time()
    
    losses_train, acc_train = [],[]
    losses_val, acc_val = [],[]

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

        loss = train_epoch(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        #loss, acc = train_epoch(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        losses_train.append(loss)
        #acc_train.append(acc)

        print(f'    Time: {timeSince(start, epoch / config.epochs)}, '
              f'Epochs completed: {epoch / config.epochs * 100}%, '
              f'Epoch loss: {loss:.4f}')
            #f'Epoch accuracy: {acc:.4f}')


        #wandb.log({'epoch': epoch, 'train/loss': avg_epoch_loss, 'train/accuracy': avg_epoch_acc})

        # Validation
        encoder.eval()
        decoder.eval()

        with torch.no_grad():

            print(f'\n   Validation: epoch {epoch}')
            
            val_loss = val_epoch(val_loader, encoder, decoder, criterion, input_lang, output_lang)
            #val_loss, val_acc = val_epoch(val_loader, encoder, decoder, criterion, input_lang, output_lang)

            losses_val.append(val_loss)
            #acc_val.append(val_acc)

            #wandb.log({'epoch': epoch, 'validation/loss': avg_val_loss, 'validation/accuracy': avg_val_acc})
        
        encoder.train()
        decoder.train()



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



# MAIN
def train(input_lang, output_lang, train_loader, val_loader):
    # Create encoder and decoder
    encoder = EncoderRNN(input_lang.n_words, config.latent_dim).to(device)
    decoder = DecoderRNN(config.latent_dim, output_lang.n_words).to(device)
    # Save models
    torch.save(encoder.state_dict(), config.encoder_path)
    torch.save(decoder.state_dict(), config.decoder_path)

    print("Encoder and decoder created.\n")

    # Train the decoder and encoder
    trainSeq2Seq(train_loader, val_loader, encoder, decoder, input_lang, output_lang)
    print("\nModel trained successfully.")

    #evaluateRandomly(encoder, decoder)
