#-------------------------------------------#
#--------------CONFIGURATION----------------#
#-------------------------------------------#

batch_size = 128  # Batch size for training.
epochs = 2  # Number of epochs to train for.
latent_dim = 256 #1024 # Latent dimensionality of the encoding space.
num_samples =  90000 # Number of samples to train on.
input_dim = 81 #num_encoder_tokens - number of unique char of language 1
output_dim = 99 #num_decoder_tokens - number of unique char of language 2
validation_split = 0.2
learning_rate = 0.0001
cell_type = 'LSTM' #'GRU'
ltsm_layers = 1
dropouts = 0
opt = 'RMSprop' #'adam'
name = "Execution"

# Path to the data txt file on disk.
# './cat-eng/cat.txt' el dataset en catala nomes te 1336 linies
data_path = './spa-eng/spa.txt' #139705 lines
encoder_path='./models/encoder_modelTranslation.h5'
decoder_path='./models/decoder_modelTranslation.h5'
char2encoding_path = './models/char2encoding.pkl'
model_encoder_path = './models/model_encoder.png'
model_decoder_path = './models/model_decoder.png'
full_model_path = './models/model.png'
