[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14935852&assignment_repo_type=AssignmentRepo)
# Machine Translation
Automatic machine translation model from one sentence to another.

Our goal is to create a translation model that can convert sentences from English-Spanish and Spanish-English. The model is based in RNN and have two option of cell types: LSTM and GRU.

Here is an image to have an idea of what the model does:
![Model Image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/images/image_model.png?raw=true)

## Code structure
* images

    It contains a photo of how the encoder-decoder model works for both the word-based model, and for the character-based model. 
    
    It also have the images of the plots obtained in the plots.py.

* models

    It stores the model encoder.h5 and decoder.h5 that was last run. It also contains CSV files with the translations that the last model run gave, for all test, train and validation datasets. It contains a subfolder to play telephone (play_telephone.py) with the best encoders and decoders for both languages Spanish and English. On a previous version, the folder contained subfolders for each language (eng-spa and spa-eng) and also in each language contained subfolders with each cell type (LSTM and GRU). However, this folder was simplified because storing that many models made the folder weight 1GB.

* spa - eng

    It contains the raw data, composed by some sentences in Spanish and English.

* utils

    The folder has four .py inside.
    - data.py is used to preprocess the data in order to enter it to the model
    - plots.py make plots of the hyperparameter exploration (can only be run if some specific folders are in the directory - we didn't include them in the final delivery because they were too heavy.)
    - training.py is the loop of training the model
    - testing.py is the loop of testing the model
    
    For more specific and technical information, go to *MODEL_SPECIFICATIONS.md* or follow this [link](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/MODEL_SPECIFICATIONS.md).
    
    Also, if you have curiosity about the hyperparameters we chose, in this [link](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/HYPERPARAMETER_EXPLORATION.md) you have all the process we made (or go to the file HYPERPARAMETER_EXPLORATION.md).

* wandb

    The plots and figures of the data when the model is trained is stored in order for the Weight and Bias web application to plot the data. This folder is only shown if config.do_wandb is set to True in config.py.

* config.py

    The parameters of our model are initialized

* environment.yml

    The dependencies necessary to run our project

* main.py

    The main file, where all the project would be executed

* play_telephone.py

    The main idea of this file is to compute a transcription from one language to the other and, with the result obtained, do the inverse transcription in order to see how the model performs. In order to play, some specific configurations are need to be set on the config.py file:
  - config.model has to be set to "words".
  - config.reverse must be set to False.
  - config.max_lenght must be set to 15.
  - config.latent_dim must be set to 256.
  - config.cell_type must be set to LSTM.
  
  This is because, in the original version of play_telephone.py, it took the latest models run from Spanish-English and English-Spanish stored in the models folder. However, since we had to delete the models stored for the final delivery (again, the file was too heavy - each model stored occupied 50MB), we decided to set it so that it can be used with some determined models. That's why the configurations in config.py must be aligned with the models' configurations.
 

## How To Use
To clone and run this project, you will need Git and Conda installed on your computer.
An alternative of conda is have installed the packages on the environment.yml file.

If you have Conda installed, from your command line write the following instructions to create an environment with all the dependencies and then activate it:

```bash
# Clone this repository
$ git clone https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05

# Go into the repository
$ cd deep-learning-project-2024-ai_nndl_group_05

# Creating the environment with the provided environment.yml file
$ conda env create --file environment.yml

# Activating the environment
$ conda activate projectDL

# For running the project
py main.py
```

If you have trouble activating the environment and the error that shows up says something like:
    -  Your shell has not been properly configured to use 'conda activate'

Do this instructions in the terminal:
```bash
# To know which type of shell you have 
$ echo $SHELL

# IF the output of the command above is for example: /bin/bash  

# To initialize the shell 
$ conda init bash

# To activate the environment
$ conda activate projectDL
```

To select the interpreter of the environment, press 'Ctrl+Shift+P', search 'Python: Select interpreter' and click on the environment interpreter.
If you can't see the interpreter, in the terminal write
```
which python
```
Copy the path that output. Press again 'Ctrl+Shift+P', press in 'Enter interpreter path' and paste the path you have copied before.



In case that you can't see all the branches, run the command:
```bash
git fetch --all
```


## Contributors
Sara Martín Núñez -- Sara.MartinNu@autonoma.cat
Lara Rodríguez Cuenca -- Lara.RodriguezC@autonoma.cat
Iván Martín  Campoy -- ivan.martinca@autonoma.cat
Aina Navarro Rafols -- Aina.NavarroR@autonoma.cat

Neural Network and Deep Learning
Degree in Artificial Intelligence
UAB, 2023-24
