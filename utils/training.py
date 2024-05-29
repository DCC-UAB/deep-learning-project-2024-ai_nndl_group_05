from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import wandb
import jiwer
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
def compute_accuracy(predictions, targets, output_lang, eos_token="EOS"):

    def tensor_to_words(tensor, lang, eos_token):
        words = []
        for idx in tensor:
            word = lang.index2word[idx.item()]
            if word == eos_token:
                break
            words.append(word)
        return words
    
    def tensor_to_chars(tensor, lang, eos_token):
        chars = []
        for idx in tensor:
            char = lang.index2char[idx.item()]
            if idx == eos_token:
                break
            chars.append(char)
        return chars


    batch_size = predictions.size(0)
    total_correct = 0
    total_words = 0
    
    for i in range(batch_size):
        predicted_ids = predictions[i].max(dim=-1)[1]  # Get the predicted word indices
        
        if config.model == "words":
            reference_words = tensor_to_words(targets[i], output_lang, eos_token)
            predicted_words = tensor_to_words(predicted_ids, output_lang, eos_token)
        
            for pred_word, ref_word in zip(predicted_words, reference_words):
                if pred_word == eos_token or ref_word == eos_token:
                    break
                if pred_word == ref_word:
                    total_correct += 1
                total_words += 1

        elif config.model == "chars":
            reference_chars = tensor_to_chars(targets[i], output_lang, eos_token)
            predicted_chars = tensor_to_chars(predicted_ids, output_lang, eos_token)
            
            for pred_char, ref_char in zip(predicted_chars, reference_chars):
                if pred_char == "EOS" or ref_char == "EOS":
                    break
                if pred_char == ref_char:
                    total_correct += 1
                total_chars += 1

    if config.model == "words":
        accuracy = total_correct / total_words if total_words > 0 else 0
    elif config.model == "chars":
        accuracy = total_correct / total_chars if total_chars > 0 else 0

    return accuracy

def wer(reference, hypothesis, eos_token="EOS"):
    total_error = 0
    total_words = 0
    if reference != eos_token and hypothesis != eos_token:  
        error = jiwer.wer(reference, hypothesis)
        total_words += len(reference.split())
        total_error += error
    wer_value = total_error / total_words if total_words > 0 else 0
    return wer_value

def per(reference, hypothesis, eos_token="EOS"):
    total_error = 0
    total_words = 0  
    for ref, hyp in zip(reference, hypothesis):
        ref_chars = ''.join(' '.join(ref).split(eos_token)[0])
        hyp_chars = ''.join(' '.join(hyp).split(eos_token)[0])
        error = jiwer.wer(ref_chars, hyp_chars)
        total_words += len(ref)
        total_error += error
    per_value = total_error / total_words if total_words > 0 else 0
    return per_value

def evaluate_per(predictions, targets, output_lang):
    def tensor_to_words(tensor, lang):
        words = []
        for idx in tensor:
            words.append(lang.index2word[idx.item()])
        return words

    total_per = 0
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        predicted_ids = predictions[i].max(dim=-1)[1]  
        reference_sentence = tensor_to_words(targets[i], output_lang)
        hypothesis_sentence = tensor_to_words(predicted_ids, output_lang)
        
        per_value = per(reference_sentence, hypothesis_sentence)
        total_per += per_value
    
    average_per = total_per / batch_size
    return average_per

def evaluate_wer(predictions, targets, output_lang, eos_token="EOS"):
    def tensor_to_words(tensor, lang, eos_token):
        words = []
        for idx in tensor:
            word = lang.index2word[idx.item()]
            if word == eos_token:
                break
            words.append(word)
        return words

    total_wer = 0
    batch_size = predictions.size(0)
    
    for i in range(batch_size):
        predicted_ids = predictions[i].max(dim=-1)[1] 
        reference_sentence = tensor_to_words(targets[i], output_lang, eos_token)
        hypothesis_sentence = tensor_to_words(predicted_ids, output_lang, eos_token)
        reference_sentence_str = ' '.join(reference_sentence)
        hypothesis_sentence_str = ' '.join(hypothesis_sentence)
        wer_value = wer(reference_sentence_str, hypothesis_sentence_str)
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
    
    def get_chars(lang, tensor):
        _, topi = tensor.topk(1)
        ids = topi.squeeze()
        chars = []
        for idx in ids:
            if idx.item() == EOS_token:
                chars.append('EOS')
                break
            chars.append(lang.index2char[idx.item()])
        return chars

    if config.model == "words":
        input = [input_lang.index2word[idx.item()] for idx in input_tensor if idx!=0]
        decoded = get_words(output_lang, decoded_outputs)
        target = [output_lang.index2word[idx.item()] for idx in target_tensor if idx!=0]
    elif config.model == "chars":
        input = [input_lang.index2char[idx.item()] for idx in input_tensor]
        decoded = get_chars(output_lang, decoded_outputs)
        target = [output_lang.index2char[idx.item()] for idx in target_tensor]

    return input, decoded, target


# ENCODER / DECODER
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=config.dropouts):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        
        if config.cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif config.cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)

        if config.cell_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        elif config.cell_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

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
        output, hidden = self.rnn(output, hidden)
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
        acc = compute_accuracy(decoder_outputs, target_tensor, output_lang)
        total_acc += acc

        if config.model == "words":
            wer = evaluate_wer(decoder_outputs, target_tensor, output_lang)
            total_wer += wer 
            per = evaluate_per(decoder_outputs, target_tensor, output_lang)
            total_per += per  

        if batch_idx % config.batch_size == 0:
            if config.model == "words":
                print(f'    Step [{batch_idx+1}/{len(dataloader)}], ' 
                    f'Loss: {loss.item():.4f}, '
                    f'Accuracy: {acc:.4f}, '
                    f'WER: {wer:.4f}, '
                    f'PER: {per:.4f}')
            elif config.model == "chars":
                print(f'    Step [{batch_idx+1}/{len(dataloader)}], ' 
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {acc:.4f} ')

    average_loss = total_loss / len(dataloader)  # Calculate average loss over all batches
    average_acc = total_acc / len(dataloader)   # Calculate average accuracy over all batches
    
    if config.model == "words":
        average_wer = total_wer / len(dataloader)  # Calculate average WER over all batches
        average_per= total_per / len(dataloader)  # Calculate average PER over all batches
    
        return average_loss, average_acc, average_wer, average_per
    
    elif config.model == "chars":
        return average_loss, average_acc, None, None


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

        acc = compute_accuracy(decoder_outputs, target_tensor, output_lang)

        total_loss += loss.item()
        total_acc += acc

        if config.model == "words":
            # Compute WER for the current batch
            wer = evaluate_wer(decoder_outputs, target_tensor, output_lang)
            total_wer += wer  # Sum the batch-wise average WER

            per = evaluate_per(decoder_outputs, target_tensor, output_lang)
            total_per += per  # Sum the batch-wise BLEU score

        if batch_idx % config.batch_size == 0:
            if config.model == "words":
                print(f'        Step [{batch_idx+1}/{len(dataloader)}], ' 
                    f'Loss: {loss.item():.4f}, '
                    f'Accuracy: {acc:.4f}, '
                    f'WER: {wer:.4f}, '
                    f'PER: {per:.4f}')
            elif config.model == "chars":
                print(f'        Step [{batch_idx+1}/{len(dataloader)}], ' 
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {acc:.4f} ')

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

    if config.model == "words":
        average_wer = total_wer / len(dataloader) 
        average_per = total_per/len(dataloader)
        return average_loss, average_acc, average_wer, average_per
    
    elif config.model == "chars":
        return average_loss, average_acc, None, None


# TRAINING LOOP

def trainSeq2Seq(train_loader, val_loader, encoder, decoder,
                 input_lang, output_lang):
    
    start = time.time()
    
    losses_train, acc_train, wer_train, per_train = [],[],[],[]
    losses_val, acc_val, wer_val, per_val = [],[],[],[]

    print(f"Cell type: {config.cell_type}")
    print(f"Hidden dimensions: {config.latent_dim}\n")
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

        loss, acc, wer, per = train_epoch(train_loader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang, n_epoch=epoch)

        losses_train.append(loss)
        acc_train.append(acc)
        wer_train.append(wer)
        per_train.append(per)

        if config.model == "words":
            print(f'    Time: {timeSince(start, epoch / config.epochs)}, '
                f'Epochs completed: {epoch / config.epochs * 100}%, '
                f'Epoch loss: {loss:.4f}, '
                f'Epoch accuracy: {acc:.4f}, '
                f'Epoch WER: {wer:.4f}, '
                f'Epoch PER: {per:.4f}')
        elif config.model == "chars":
            print(f'    Time: {timeSince(start, epoch / config.epochs)}, '
              f'Epochs completed: {epoch / config.epochs * 100}%, '
              f'Epoch loss: {loss:.4f}, '
              f'Epoch accuracy: {acc:.4f}')

        if config.do_wandb:
            if config.model == "words":
                wandb.log({'epoch': epoch, 'train/loss': loss, 'train/accuracy': acc, 'train/WER': wer, 'train/PER': per})
            elif config.model == "chars":
                wandb.log({'epoch': epoch, 'train/loss': loss, 'train/accuracy': acc})


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

            if config.do_wandb:
                if config.model == "words":
                    wandb.log({'epoch': epoch, 'validation/loss': val_loss, 'validation/accuracy': val_acc, 'validation/WER': val_wer, 'validation/PER': val_per})
                elif config.model == "chars":
                    wandb.log({'epoch': epoch, 'validation/loss': val_loss, 'validation/accuracy': val_acc})
                    
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
