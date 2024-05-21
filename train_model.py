import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #dssable info messages

import wandb

import config
from utils.data import prepareData
from utils.training import train



config_defaults = {
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "dataset": config.data_path,
        "epochs": config.epochs,
        "latent_dim": config.latent_dim,
        "cell_type": config.cell_type,
        "optimizer":  config.opt,
        "lstm_layers": config.ltsm_layers, 
        'dropouts': config.dropouts
        }


def create_wandb():
    wandb.init(
        # set the wandb project where this run will be logged
        project="Machine_Translation",
        # track hyperparameters and run metadata
        config=config_defaults,
        name = "Execution",
        allow_val_change=True)


if __name__ == "__main__":
    
    # start a new wandb run to track this script
    wandb.login(key="8090840539532ccc2f8ea5c1595fde6dbb57bf56")
    create_wandb()

    # Data preprocessing
    print("\n#----------------------------------------#")
    print("-------STARTING DATA PROCESSING------------")
    print("#----------------------------------------#\n")
    
    encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length, encoder_dataset, decoder_input_dataset, decoder_target_dataset=prepareData(config.data_path)
    
    # Training the model
    print("\n#----------------------------------------#")
    print("-------STARTING MODEL TRAINING------------")
    print("#----------------------------------------#\n")
    train(encoder_input_data, decoder_input_data, decoder_target_data, input_token_index, target_token_index,input_texts,target_texts,num_encoder_tokens,num_decoder_tokens,max_encoder_seq_length, encoder_dataset, decoder_input_dataset, decoder_target_dataset)