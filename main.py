import wandb
import torch

import config
from utils.data import get_dataloader
from utils.training import train
from utils.testing import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        name = config.name,
        allow_val_change=True
        )


if __name__ == "__main__":
    
    # start a new wandb run to track this script
    wandb.login(key="8090840539532ccc2f8ea5c1595fde6dbb57bf56")
    create_wandb()

    # Data preprocessing
    print("\n#----------------------------------------#")
    print("-------STARTING DATA PROCESSING------------")
    print("#----------------------------------------#\n")
    
    input_lang, output_lang, train_loader, val_loader, test_loader = get_dataloader()
    
    # Training the model
    print("\n#----------------------------------------#")
    print("-------STARTING MODEL TRAINING------------")
    print("#----------------------------------------#\n")
    
    train(input_lang, output_lang, train_loader, val_loader)

    print("\n#----------------------------------------#")
    print("--------STARTING MODEL TESTING-------------")
    print("#----------------------------------------#\n")

    # Get test dataset accuracy
    test(input_lang, output_lang, test_loader, type='test')
    
    # Get training and validation datasets final translations
    test(input_lang, output_lang, train_loader, type='train')
    test(input_lang, output_lang, val_loader, type='val')
    