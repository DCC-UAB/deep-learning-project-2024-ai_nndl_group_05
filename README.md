[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jPcQNmHU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=14935852&assignment_repo_type=AssignmentRepo)
# XNAP-Machine Translation
Automatic machine translation model from one text to another

Our goal is to create a translation model that can convert text from english-spanish and spanish-english. The model is based in RNN and have two  option of cell types: LSTM and GRU



## Code structure
* Models

    It has photos of the Encoder and Decoder part of our model.  It also has, for each translation (eng-spa and spa-eng) and for each type of model (LSTM and GRU) some csv files with the data preprocessed and extracted from the dataset before entering the model.

* Spa - eng

    It containts the raw data, composed by some sentences in spanish and english.

* utils

    The folder has three .py inside.
    - data.py is used to preprocess the data in order to enter to the model
    - training.py is the loop of training the model
    - test.py is the loop of testing the model
    For more specific and technical information, go to *MODEL_SPECIFICATIONS.md* or follow this [link](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/MODEL_SPECIFICATIONS.md).

* wandb

    The plots and figures of the data when the model is trained is stored in order for the Weight and Bias web application to plot the data.

* congif.py

    The parameters of our model are initialized

* environment.yml

    The dependencies necessaries to run our project

* main.py

    The main file, where all the project would be executed

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

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
python main.py
```


## Contributors
Sara Martín Núñez -- Sara.MartinNu@autonoma.cat
Lara Rodríguez Cuenca -- Lara.RodriguezC@autonoma.cat
Iván Martín  Campoy -- ivan.martinca@autonoma.cat
Aina Navarro Rafols -- Aina.NavarroR@autonoma.cat

Neural Network and Deep Learning
Degree in Artificial Intelligence
UAB, 2023-24
