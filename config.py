#-------------------------------------------#
#--------------CONFIGURATION----------------#
#-------------------------------------------#

model = "chars" #"words"
do_wandb = False # Recommended to False.

# Set language
reverse = False

if reverse == False:
    language = "eng-spa"
    input_language = 'eng'
    output_language = 'spa'

if reverse == True:
    language = "spa-eng"
    input_language = 'spa'
    output_language = 'eng'

# Set wandb model name
if model == "words":
    if reverse == False:
        project = "Machine_Translation_words"
    elif reverse == True:
        project = "Machine_Translation_words_2"
    
    max_length = 15 # Max number of words in sentence: 15 - recommended.

elif model == "chars":
    if reverse == False:
        project = "Machine_Translation_chars"
    elif reverse == True:
        project = "Machine_Translation_chars_2"

    max_length = 20 # Max number of chars in sentence: 40 - recommended.


# Seq2Seq architecture
cell_type = 'LSTM' #'LSTM' or 'GRU'
latent_dim = 256 #128 or 256 # Latent dimensionality of the encoding space.

# Datasets
validation_split = 0.2
test_split = 0.2

# Training process
batch_size = 64  # Batch size for training.
epochs = 3  # Number of epochs to train for. 20 recommended.
learning_rate = 0.001 #0.001 (recommended) or 0.01
criterion = 'NLLLoss' #'CrossEntropyLoss'
ltsm_layers = 2
dropouts = 0.2
opt = 'Adam' #'Adam' or 'RMSprop'
name = f'{cell_type}-latent_dim={latent_dim},{criterion},opt={opt},lr={learning_rate},dropout={dropouts}'


# Paths
path = f'{model}/{language}/{cell_type}'
data_path = './spa-eng/spa.txt' #139705 lines


save_models = False # Recommended to False.

# If save_models == True: this paths must exists:

if save_models: # Paths to distinguish between different models.
    # Models
    encoder_path = f'./models/{path}/encoder.h5'
    decoder_path =f'./models/{path}/decoder.h5'

    # Translation csv
    results_path_train = f'./models/{path}/traindata_translations.csv'
    results_path_val = f'./models/{path}/valdata_translations.csv'
    results_path_test = f'./models/{path}/testdata_translations.csv'

else: # Simple path
    encoder_path = f'./models/encoder.h5'
    decoder_path =f'./models/decoder.h5'
    results_path_train = f'./models/traindata_translations.csv'
    results_path_val = f'./models/valdata_translations.csv'
    results_path_test = f'./models/testdata_translations.csv'
