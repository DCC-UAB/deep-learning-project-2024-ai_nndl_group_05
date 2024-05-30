import random
from utils.data import get_dataloader
from utils.training import *
import config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadModels(input_lang, output_lang):
    encoder_io = EncoderRNN(input_lang.n_words, config.latent_dim).to(device)
    decoder_io =  DecoderRNN(config.latent_dim, output_lang.n_words).to(device)
    encoder_io.load_state_dict(torch.load('./models/best_models/encoder_eng-spa.h5', map_location=torch.device('cpu')))
    decoder_io.load_state_dict(torch.load('./models/best_models/decoder_eng-spa.h5', map_location=torch.device('cpu')))
    
    encoder_oi = EncoderRNN(output_lang.n_words, config.latent_dim).to(device)
    decoder_oi =  DecoderRNN(config.latent_dim, input_lang.n_words).to(device)
    encoder_oi.load_state_dict(torch.load('./models/best_models/encoder_spa-eng.h5', map_location=torch.device('cpu')))
    decoder_oi.load_state_dict(torch.load('./models/best_models/decoder_spa-eng.h5', map_location=torch.device('cpu')))

    return encoder_io, decoder_io, encoder_oi, decoder_oi

def get_random_batch(data_loader):
    # Get the number of batches
    num_batches = len(data_loader)
    # Select a random batch index
    random_batch_idx = random.randint(0, num_batches - 1)
    
    # Iterate through the DataLoader to find the batch at random_batch_idx
    for batch_idx, data in enumerate(data_loader):
        if batch_idx == random_batch_idx:
            input_tensor, target_tensor = data
            return input_tensor, target_tensor


def get_ids(decoder_outputs):

    all_ids = []
    
    for tensor in decoder_outputs: 
        _, topi = tensor.topk(1)
        ids_ = topi.squeeze()

        ids = []
        for id in ids_:
            if id == 1:
                ids.append(1)
                while len(ids) < config.max_length:
                    ids.append(0)
                break
            else:
                ids.append(id.item())
        
        all_ids.append(ids)
    
    return torch.tensor(all_ids).to(device)
  

def telephone(input_lang, output_lang, encoder, decoder, data_loader, 
              way=1, output_tensor=None, first_input=None, shown_sentences=20):

    encoder.eval()
    decoder.eval()

    criterion = {'NLLLoss': nn.NLLLoss, 'CrossEntropyLoss': nn.CrossEntropyLoss}[config.criterion]()
    
    translated_sentences = []

    with torch.no_grad():

        # Get input and target tensor
        if way == 1:    
            input_tensor, target_tensor = get_random_batch(data_loader)

        elif way == 2:
            input_tensor, target_tensor = output_tensor, first_input
        
        input_tensor.to(device), target_tensor.to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # Compute metrics
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        acc = compute_accuracy(decoder_outputs, target_tensor, output_lang)
        wer = evaluate_wer(decoder_outputs, target_tensor, output_lang)

        for i, (input, output, target) in enumerate(zip(input_tensor, decoder_outputs, target_tensor)):
            input_words, decoded_words, target_words = translate(input_lang, output_lang, 
                                                                    input, output, target)
            translated_sentences.append((input_words, decoded_words, target_words))

            print(f'    {input_lang.name}: {input_words}')
            print(f'    {output_lang.name} translation: {decoded_words}')
            print(f'    {output_lang.name} ground truth: {target_words}\n')

            if i == shown_sentences:
                break

    if way == 1:
        return get_ids(decoder_outputs), input_tensor
    elif way == 2:
        return get_ids(decoder_outputs), _


if __name__ == "__main__":

    if (config.model == "chars"):
        print("config.model must be set to 'words' to play.")
    if (config.reverse == True):
        print("config.reverse must be set on False to play.")
    if (config.latent_dim != 256):
        print("config.latent_dim must be set to 256 to play.")
    if (config.cell_type != "LSTM"):
        print("config.cell_type must be set to LSTM to play.")
    if (config.max_length != 15):
        print("config.max_lenght must be set to 15 to play.")
    else:
        print("\n#----------------------------------------#")
        print("-------GETTING TEST DATALOADER------------")
        print("#----------------------------------------#\n")
        # Get random pair of sentences from test loader
        input_lang, output_lang, train_loader, val_loader, test_loader = get_dataloader()

        print("\n#----------------------------------------#")
        print("-------STARTING TELEPHONE GAME------------")
        print("#----------------------------------------#\n")

        print("Playing telephone from Spanish to English and English to Spanish!\n")

        # Import models
        encoder_io, decoder_io, encoder_oi, decoder_oi = loadModels(input_lang, output_lang)
        print("Models imported.\n")

        print("#--------WAY ONE: ENGLISH TO SPANISH--------#\n")
        # Get translation of sentence of input language
        output_tensor, input_tensor = telephone(input_lang, output_lang, encoder_io, decoder_io, test_loader, 
                                                way=1, shown_sentences=20)

        print("#--------WAY TWO: SPANISH TO ENGLISH--------#\n")
        # Get translation of sentence of output language
        output_tensor2, _ = telephone(output_lang, input_lang, encoder_oi, decoder_oi, test_loader, 
                                    2, output_tensor, input_tensor, shown_sentences=20)
