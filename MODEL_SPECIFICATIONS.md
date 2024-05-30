Here is the explanation in depth of our model structure.
The files *data.py*, *testing.py* and *training.py* are the ones explained below:

## data.py
In data.py we deal with data preprocessing, for that we define several functions. First, we create a python class, called Lang, that, given a name, it creates a language which will be initialized with different dictionaries, so we can easily pass from word to index, from index to word and from word to count. 

To achieve that, we implement two methods in the Lang class, ‘addSentence’ and ‘addWord’. ‘addWord’ is used for every word detected in a sentence. In the method, each word gets stored in the dictionaries mentioned before (with distinct numbers for each word). We also consider the tokens  ‘start-of-string’ and ‘end-of-string’  which have assigned numbers 0 and 1 respectively.

Then we created functions to normalize the data, in this case, lower all the input words, and remove non-letter characters.

After that, we made a function to read the data, it receives two languages as input, and defines the first language as the input language, and the second as the output language unless in the config file, reverse it is marked as “True”. To compare the input to the correct translation, we separate each line from the dataset, and define the pairs, and filter those sentences that have a length larger than what we defined in the configuration file as the limit.

Finally, we create functions that will help us with the data loader, the first function, “indexesFromSentence”, adds each word to the ‘word2index’ dictionary with the respective value. The other function is ‘prepareData’, which calls the different functions we have defined previously and given two languages, returns the input language, the output language and the pairs. Finally, in the main, we call all the functions and create a dataloader for the train, validation and test set, and return both languages and the dataloader, then we move to the training file.



## testing.py
In this file, we have all the functions for testing our model once trained.

The first function, called load ’EncoderDecoderModel’, receives input_lang and output_lang as input. Then, inside itself, we create the encoder and decoder with the characteristics that the user established at the beginning of the project, and once the model is created with its parameters, it’s retrieved from the library of models (once the model is trained, is stored in one of our folders, models).

The last function is called test, and consist of calling the function explained above, then initialize the criterion and compute the test process:
- Obtain input and target tensor
- Call the model to predict for the test data
- Compute the metrics accuracy, wer and per
- Call the function translate (inside the training.py file) to obtain the sentences and not the numbers
- Print the metrics with the loss for each batch
- Print the average of all the metrics in the previous step at the end of the test 
- Store the results and metrics obtained
- Store the sentences (original and translated)



## training.py
In this file we have some function to compute metrics like the accuracy, wer and per. Furthermore, there are other functions to know, when training the model, how much time has been running.

The translate function, its main purpose, is, as its name says, to translate from numeral ids to words. It does it by the tensors it receives as input, it converts them into words using the word_to_ix dictionary of the corresponding language.

This file also contains the RNN model, composed by an encoder and decoder:
#### ENCODER
- Initialization (__init__ method):
    Parameters:

        - input_size: The size of the input vocabulary.

        - hidden_size: The size of the hidden state in the GRU/LSTM.
        
        - dropout_p: Dropout probability for regularization (default taken from config.dropouts).
    Layers:

        - self.embedding: An embedding layer that converts input tokens into dense vectors of size hidden_size.
        
        - self.rnn: A GRU/LSTM layer that processes the embedded input sequence. It has input size hidden_size, output size hidden_size, and operates with batches (batch_first=True).
        
        - self.dropout: A dropout layer to apply dropout regularization to the embedded inputs.

- Forward Method (forward method):
    input: The input sequence, typically a batch of token indices.
    Process:
        - Embedding: The input indices are passed through the embedding layer to get dense vector representations.
        
        - Dropout: Dropout is applied to the embeddings to prevent overfitting.
        
        - GRU/LSTM: The processed embeddings are passed through the GRU/LSTM. The GRU/LSTM. The GRU/LSTM outputs:
            output: The output features for each time step.
            hidden: The final hidden state of the GRU/LSTM.
        
        - Output: Returns output and hidden.
In summary, the process would be:
    
    1. Embedding layer to convert input tokens to dense vectors.
    
    2. Dropout layer for regularization.
    
    3. GRU/LSTM layer to process the embedded sequence and produce hidden states.

#### DECODER
- Initialization (__init__ method):

    Parameters:
        
        - hidden_size: The size of the hidden state in the GRU/LSTM.
        
        - output_size: The size of the output vocabulary.
    
    Layers:
        
        - self.embedding: An embedding layer that converts output tokens into dense vectors of size hidden_size.
        
        - self.gru: A GRU/LSTM layer that processes the embedded input sequence. It has input size hidden_size and output size hidden_size, operating with batches (batch_first=True).
        
        - self.out: A linear layer that maps the GRU/LSTM outputs to the output vocabulary size.

- Forward Method (forward method):
    
    Inputs:
        
        - encoder_outputs: The outputs from the encoder (not used directly in this implementation).
        
        - encoder_hidden: The final hidden state from the encoder, used to initialize the decoder's hidden state.
        
        - target_tensor: The target sequence (used for teacher forcing during training).
    
    Process:
    
        - Initialization:
            
            decoder_input: Initialized with the start-of-sequence token (SOS_token) for each sequence in the batch.

            decoder_hidden: Initialized with the final hidden state from the encoder.
        
        - Decoding Loop:
            
            For each time step up to config.max_length:
                
                - forward_step: Processes the current input and hidden state to produce an output and update the hidden state.
                
                - Teacher Forcing: If target_tensor is provided, the next input is taken from the target sequence. Otherwise, the next input is the model's own prediction from the previous step.
        
        - Concatenation and Softmax:
            
            The outputs from each time step are concatenated.
            
            Log softmax is applied to the concatenated outputs to get log probabilities.
            
            Output: Returns the concatenated decoder outputs, the final hidden state, and None (placeholder for consistency).

- Forward Step Method (forward_step method):
    
    Input:

        - input: The current input token indices.

        - hidden: The current hidden state of the GRU/LSTM.

    Process:
        
        - Embedding: The input indices are passed through the embedding layer.

        - ReLU Activation: A ReLU activation is applied to the embeddings.

        - GRU/LSTM: The processed embeddings are passed through the GRU/LSTM to produce:

            output: The output features for the current time step.
            
            hidden: The updated hidden state.

        - Linear Layer: The GRU/LSTM output is passed through the linear layer to map it to the output vocabulary size.

    Output: Returns output and hidden.


In summary, the process would be:
    
    1. Embedding layer to convert target tokens to dense vectors.

    2. GRU/LSTM layer to process the embedded sequence and produce hidden states.

    3. Linear layer to map GRU/LSTM outputs to the output vocabulary.

    4. Decoding loop that handles teacher forcing and prediction generation.


There are also functions as train_epoch and val_epoch, which define the structure of each epoch. 

    TRAIN:
        
        Receive the dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, output_lang and n_epoch
	
    VAL:

        Receive the dataloader, encoder, decoder, criterion, input_lang, output_lang


The process of both functions is similar, so it can be resumed as:

    1. Obtain the data (input tensor and target tensor)

    2. Call the encoder previously created in the function train to obtain the encoder output
    
    3. Call the decoder previously created in the function train to obtain the decoder output
    
    4. Obtain the loss 

#### RELEVANT CONCRETE PART OF THE LSTM:

Neverthless is important to consider the following part of our code:

	if target_tensor is not None:
	                #Teacher forcing: Feed the target as the next input
	                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
	            else:
	                #Without teacher forcing: use its own predictions as the next input
	                _, topi = decoder_output.topk(1)
	                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

This actually refers that our lstm to avoid spending lot of time learning from scratch the correct words it uses as input the ground truth and only predicts the next word (without this one affecting the next one, as it will take the ground truth)


#### EVALUATION METHODS
Lastly, is important to consider the different evaluate methods that were used in character level and word level:

- Both of them use accuracy and loss evaluation. While loss can be determinant to know how the model is evolving over the time, the acuracy is based on the obtained sequences (comparing the expected and the predicted). As in this case, each sentence only has one correct output (so no synonyms are cosidered correct), just having one word wrong across the sentence will decay a lot the accuracy value. Taht is why we have tried other mechanisms:
-  For word level implementation WER (Word Error Rate) has been used. This mechanism may also be referred to as the length normalized edit distance. It uses insertion, deletions and subtractions (generally substractions is 2 times more costly than the other actions) and the sum of these actions is divided by the number of words. It is said that a WER between 5-10% is considered readable [click the [LINK](https://www.futurebeeai.com/blog/breaking-down-word-error-rate) to go to the page].
-  For character level CER (Character Error Rate) it actually works in the same way than WER but for character level. In this case, a good CER for printed text is between 0.5 and 2%, while for handwritten text, it ranges between 2 and 8% [click the [LINK](https://help.transkribus.org/character-error-rate-and-learning-curve) to go to the page].
- However this metrics provide no details on the nature of translation errors and further work is therefore required to identify the main source(s) of error and to focus any research effort.

