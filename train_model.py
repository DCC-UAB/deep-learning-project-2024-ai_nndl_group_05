import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #dssable info messages

import wandb

from utils.data import prepareData
from utils.training import train

#-----------------GLOBAL VARIABLES-------------------#

batch_size = 128  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256 #1024 # Latent dimensionality of the encoding space.
num_samples =  90000 # Number of samples to train on.
validation_split = 0.2
learning_rate = 0.0001

LOG_PATH='./log'
# Path to the data txt file on disk.
# './cat-eng/cat.txt' el dataset en catala nomes te 1336 linies
data_path = './spa-eng/spa.txt' #139705 lines
encoder_path='./models/encoder_modelTranslation.h5'
decoder_path='./models/decoder_modelTranslation.h5'

name = "Execution"
opt = 'rmsprop' #'adam'

config_defaults = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dataset": data_path,
        "epochs": epochs,
        "latent_dim": latent_dim,
        "cell_type": 'LSTM', #'GRU'
        "optimizer":  opt,
        "lstm_layers": 1, 
        'dropouts': 0
        }

# ------------------FUNCTIONS-------------------#

def create_wandb():
    wandb.init(
        # set the wandb project where this run will be logged
        project="Machine_Translation",
        # track hyperparameters and run metadata
        config=config_defaults,
        name = name,
        allow_val_change=True)

# -------------------MAIN-----------------------#

if __name__ == "__main__":
    
    # start a new wandb run to track this script
    wandb.login(key="8090840539532ccc2f8ea5c1595fde6dbb57bf56")
    create_wandb()

    # Data preprocessing
    print("\n#----------------------------------------#")
    print("-------STARTING DATA PROCESSING------------")
    print("#----------------------------------------#\n")
    
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length, encoder_dataset, decoder_input_dataset, decoder_target_dataset=prepareData(data_path)
    
    # Training the model
    print("\n#----------------------------------------#")
    print("-------STARTING MODEL TRAINING------------")
    print("#----------------------------------------#\n")
    train(encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length, encoder_dataset, decoder_input_dataset, decoder_target_dataset)