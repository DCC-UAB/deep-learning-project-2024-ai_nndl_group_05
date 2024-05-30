[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14935852&assignment_repo_type=AssignmentRepo)
# Machine Translation
Automatic machine translation model from one sentence to another.

Our goal is to create a translation model that can convert sentences from english-spanish and spanish-english. The model is based in RNN and have two option of cell types: LSTM and GRU.

Here is an image to have an idea of what the model does:
![Model Image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/images/image_model.png?raw=true)

## Code structure
* images

    It contain a photo of how the encoder-decoder model works for both the word-based model, and for the character-based model. 
    
    It also have the images of the plots obtained in the plots.py

* models

    It has photos of the Encoder and Decoder part of our model.  It also has, for each translation (eng-spa and spa-eng) and for each type of model (LSTM and GRU) some csv files with the translated sentences after passing through the model.

* spa - eng

    It containts the raw data, composed by some sentences in spanish and english.

* utils

    The folder has four .py inside.
    - data.py is used to preprocess the data in order to enter it to the model
    - plots.py make plots of the hyperparameter exploration
    - training.py is the loop of training the model
    - testing.py is the loop of testing the model
    
    For more specific and technical information, go to *MODEL_SPECIFICATIONS.md* or follow this [link](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/both/MODEL_SPECIFICATIONS.md).
    
    Also, if you have curiosity for the hyperparameters we chose, in this [link](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/both/HYPERPARAMETER_EXPLORATION.md) you have all the process we made (or go to the file HYPERPARAMETER_EXPLORATION.md).

* wandb

    The plots and figures of the data when the model is trained is stored in order for the Weight and Bias web application to plot the data.

* config.py

    The parameters of our model are initialized

* environment.yml

    The dependencies necessary to run our project

* main.py

    The main file, where all the project would be executed

* play_telephone.py

    The main idea of this file is compute a transcription from one language to the other and, with the result obtained, do the inverse transcription in order to see how the model performs.
 

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
$ conda activate xnap-example

# For running the project
py main.py
```

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
