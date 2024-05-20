from __future__ import print_function
from keras.models import load_model
import numpy as np
import _pickle as pickle
import wandb

#-----------------GLOBAL VARIABLES-------------------#

FILENAME = "./output/char2encoding.pkl"

#--------------------FUNCTIONS----------------------#

def getChar2encoding(filename):
    f = open(filename, "rb")
    input_token_index = pickle.load(f)
    max_encoder_seq_length = pickle.load(f)
    num_encoder_tokens = pickle.load(f)
    reverse_target_char_index = pickle.load(f)
    num_decoder_tokens = pickle.load(f)
    target_token_index = pickle.load(f)
    f.close()
    return input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index

"""
def loadEncoderDecoderModel():
# We load the encoder model and the decoder model and their respective weights
    encoder_model= load_model(encoder_path)
    decoder_model= load_model(decoder_path)
    return encoder_model,decoder_model
"""

def decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index):
# We run the model and predict the translated sentence
    if wandb.config.cell_type == 'LSTM':
        # We encode the input
        states_value = encoder_model.predict(input_seq)

        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        
        target_seq[0, 0, target_token_index['\t']] = 1.

        stop_condition = False
        decoded_sentence = ''
        # We predict the output letter by letter 
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # We translate the token in hamain language
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # We check if it is the end of the string
            if (sampled_char == '\n' or
            len(decoded_sentence) > 500):
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            states_value = [h, c]
            
    elif wandb.config.cell_type == 'GRU':
    
        # We encode the input
        states_value = encoder_model.predict(input_seq)

        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        
        target_seq[0, 0, target_token_index['\t']] = 1.

        stop_condition = False
        decoded_sentence = ''
        # We predict the output letter by letter 
        while not stop_condition:
            output_tokens, states_value = decoder_model.predict(
                [target_seq] + [states_value])

            # We translate the token in hamain language
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # We check if it is the end of the string
            if (sampled_char == '\n' or
            len(decoded_sentence) > 500):
                stop_condition = True

            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

    return decoded_sentence

def encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens):
    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens),dtype='float32')
    for t, char in enumerate(sentence):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    return encoder_input_data

"""
# BLEU SCORE
def get_blue_score(filename):
    with open(str(filename), 'r') as f3:
        lines = f.readlines()
    for line in lines:
        element = line.strip().split('\t')
        y_true.append(element[0])
        y_pred.append(element[1])

    for y_true, y_pred in zip(y_true, y_pred):
        score += int(sentence_bleu(y_true, y_pred))
    return score
"""

#--------------------MAIN----------------------#

def test(sentence):
    input_token_index,max_encoder_seq_length,num_encoder_tokens,reverse_target_char_index,num_decoder_tokens,target_token_index= getChar2encoding(FILENAME)

    encoder_input_data=encodingSentenceToPredict(sentence,input_token_index,max_encoder_seq_length,num_encoder_tokens) 
    encoder_model= load_model('encoder_modelPredTranslation.h5')
    decoder_model= load_model('decoder_modelPredTranslation.h5')

    input_seq = encoder_input_data

    decoded_sentence=decode_sequence(input_seq,encoder_model,decoder_model,num_decoder_tokens,target_token_index,reverse_target_char_index)
    print('-')
    print('Input sentence:', sentence)
    print('Decoded sentence:', decoded_sentence)


if __name__ == "__main__":
    #test(input("Enter a sentence to translate: "))
    sentence = "What is going on?"
    test(sentence)
