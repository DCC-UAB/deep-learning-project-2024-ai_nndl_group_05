#-------------------------------------------#
#--------------CONFIGURATION----------------#
#-------------------------------------------#

language = "spa-eng"
input_language = 'eng'
output_language = 'spa'

max_length = 5

# Seq2Seq architecture
cell_type = 'GRU' #'LSTM'
latent_dim = 128 #1024 # Latent dimensionality of the encoding space.

#num_samples =  90000 # Number of samples to train on.
#input_dim = 81 #num_encoder_tokens - number of unique char of language 1
#output_dim = 99 #num_decoder_tokens - number of unique char of language 2

# Datasets
validation_split = 0.2
test_split = 0.2

# Training process
batch_size = 64  # Batch size for training.
epochs = 2  # Number of epochs to train for.
learning_rate = 0.001 #0.0001
criterion = 'NLLLoss' #'CrossEntropyLoss'
ltsm_layers = 1
dropouts = 0
opt = 'Adam' #'RMSprop'
name = f'{cell_type}-latent_dim={latent_dim},{criterion},opt={opt},lr={learning_rate},dropout={dropouts}'


# Paths

path = f'{language}/{cell_type}'

data_path = './spa-eng/spa.txt' #139705 lines

encoder_path = f'./models/{path}/encoder.h5'
decoder_path =f'./models/{path}/decoder.h5'

#char2encoding_path = f'./models/{path}/char2encoding.pkl'
#png_encoder_path = f'./models/{path}/png_encoder'
#png_decoder_path = f'./models/{path}/png_decoder'
#png_model_path = f'./models/{path}/png_model_.png'

results_path_train = f'./models/{path}/traindata_translations.csv'
results_path_val = f'./models/{path}/valdata_translations.csv'
results_path_test = f'./models/{path}/testdata_translations.csv'