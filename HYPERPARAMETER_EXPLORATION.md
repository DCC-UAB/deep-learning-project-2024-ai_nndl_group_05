# Hyperparameter exploration

We did a thorough search of hyperparameters to understand better our model and our data and see which ones maximize the accuracy and other metrics we computed, such as ‘wer’ 
(Word error rate) and ‘cer’ (Character error rate). 

Since, our time, as in, with a virtual machine is limited we couldn’t try all combinations, so we changed our approach to look what parameters got better results and then do slight modifications. We also had a hypothesis in mind, that was, hyperparameters that maximize our metrics should be similar when doing reverse translation (taking the input language as the output and the other way around). Another theory that we wanted to confirm was that if we use a different architecture, but the languages are the same, the hyperparameters would be similar.

In the repository, there’s a folder called ‘images’, we have saved both models, word-based, and character-based. In each folder, we save the plots with the different results for each pair of languages. And then compute different metrics in each set (train, validation, and test), and saving them in their respective folder as well. Each set have their own best architecture for each metric, as shown in the plots, but to have an overall vision, we will talk about the results for ‘Accuracy’, on the train and test set.

## Word-based model

On the train set, the best results when translating from English to Spanish: 

	-LSTM with a latent dimension of 256, using a Negative log likelihood Loss, an Adam optimizer, a learning rate of 0.001 and a dropout of 0. It has a 0.91 accuracy. 

 ![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/f1a642b2-897b-4087-ac46-697c0ca00741)

From Spanish to English, however the best was a 

 	-GRU, with a latent dimension of 256, NLLL loss, Adam, lr = 0.001 and dropout 0.2, we got a 0.84 training accuracy.
![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/images/words/spa-eng/train/train_acc.png?raw=true)

On the validation set, from English to Spanish:

	-LSTM with a latent dimension of 256, using a Cross Entropy Loss, an Adam optimizer, a learning rate of 0.001 and a dropout of 0.2. It has a 0.56 accuracy. 
![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/images/words/eng-spa/val/val_acc.png?raw=true)

And from Spanish to English:

	-LSTM with a latent dimension of 256, using a Negative log likelihood Loss, an RMSprop optimizer, a learning rate of 0.001 and a dropout of 0.2. It has a 0.63 accuracy. 
![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/images/words/spa-eng/val/val_acc.png?raw=true)

On the test set, when translating from English to Spanish, we found this configuration to be the best one:		

	-LSTM with a latent dimension of 256, using a Cross Entropy Loss, an Adam optimizer, a learning rate of 0.001 and a dropout of 0.2. The result was a 0.55 acuracy. 

![imagen_2024-05-29_225423790](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/5d7b24d1-5d17-4276-8edf-b45bf4b8404c)

Let's just consider why those hyperparameters might be the correct ones:

- **LSTM with a latent dimension of 256**
- **Cross Entropy Loss**
- **Adam optimizer**
- **Learning rate of 0.001**
- **Dropout of 0.2**

First the use of LSTM allows to handle sequential data and capture long-range dependencies in sequences. A latent dimension of 256 units, enhance our model with enough parameters to model the intricacies of language without being too large to cause overfitting or computational inefficiencies. 
Aside from that, We have the Cross Entropy Loss, which is suitable for classification tasks, where the goal is to predict the probability distribution of a set of discrete classes (words, in this case). It is particularly effective in measuring the performance of a model whose output is a probability value between 0 and 1. The use of Cross entropy loss helps in penalizing the divergence between the predicted probability distribution and the true distribution, thereby fine-tuning the model to improve its predictions. 
Moreover we have the optimizer: Adam (Adaptive Moment Estimation) which adjusts the learning rate for each parameter dynamically. This optimizer combines AdaGrad and RMSProp.
Beside having a learning rate of 0.01 typically provides a good balance between speed and stability of convergence, allowing the model to learn effectively without large oscillations in the loss function (low enough to ensure stable training and high enough to allow the model to make significant updates to the weights). And so it happends in the model.
Lastly, Dropout. It is a regularization technique used to prevent overfitting by randomly setting a fraction of input units to zero at each update during training. In our case the value is of 0.2 means that 20% of the neurons will be dropped out during each training step, which helps in improving the generalization of the model (not relying on specific neurons).


However, it should be mentioned that it is remarkable, that GRU goot similar results with same hyperparameters 256 latent dim, a negative log likelihood loss (NLLL), adagrad, lr = 0.001 and dropout = 0.

This is because they work similar, the main differesnce is:

If we consider the LSTM structure it has 3 gates (input, output, and forget gates) and a cell state. This allows LSTMs to learn and remember more intricate dependencies within the data. While GRU has two gates (reset and update gates) and combines the cell state and hidden state into a single vector. This makes GRUs simpler and slightly less expressive compared to LSTMs.

When translating from Spanish to English, we observed that the configuration that was the best one was:

	-LSTM with a latent dimension of 256, using a Cross Entropy Loss, an Adam optimizer, a learning rate of 0.001 and a dropout of 0.2, with a 0.64 accuracy.

Which what we considered at the start. The second-best result was not exactly what we tried, but close enough, it was a GRU with 256 latent dim, a negative log likelihood loss (NLLL), Adam , lr = 0.001 but with a dropout of 0.2 as well.

![imagen_2024-05-29_225510726](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/ad51c1eb-c69c-4edd-9d3b-314f06b377ea)



## Character-based model
On the training set, when doing Eng->Spa

	-LSTM latent dim = 256, NLLL loss, opt = RMSProp, lr of 0.001 and dropout = 0.2, accuracy = 0.78

![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/1b01f293-8be5-4955-b734-a3f0435be31a)

And from Spanish to English:

	-LSTM, latent_dim = 256, NLLL loss, opt = RMSProp, lr = 0.001, dropout = 0.2

![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/f655ebdd-0774-4222-a87c-25182dc0c9af)

On the validation set, from English to Spanish:

	-LSTM with a latent dimension of 256, using a NLLLoss, an RMSprop optimizer, a learning rate of 0.001 and a dropout of 0.2. It has a 0.78 accuracy. 
![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/images/chars/eng-spa/val/val_acc.png?raw=true)

And from Spanish to English:

	-LSTM with a latent dimension of 256, using a Negative log likelihood Loss, an RMSprop optimizer, a learning rate of 0.001 and a dropout of 0.2. It has a 0.63 accuracy. 
![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/blob/main/images/chars/spa-eng/val/val_acc.png?raw=true)

And on the test set, from English to Spanish: 

	-LSTM with a latent dimension of 256, a negative log likelihood loss, an RMSProp optimizer, a learning rate of 0.001 and a dropout of 0.2, with that, we managed to get a 0.75 accuracy.

 ![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/e0779305-5e85-43f7-be84-8f43b1041383)


And from Spanish to English:

	-LSTM with a latent dimension of 256, a negative log likelihood loss, an RMSProp optimizer, a learning rate of 0.001 and a dropout of 0.2, this had a 0.78 accuracy.

 ![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/ad05787b-18d9-4859-ba78-630d7881f98c)


For the Character-based model we can see that the hyperparameters are quite similar to the word-based ones. Only the optimizer changes from Adam to RMSProp which actually it makes sense as Adam uses RMSProp. In theory Adam is a combination of RMSprop with momentum and should be more robust and faster in convergence. However, RMSprop on its own is still very effective and might even outperform Adam in certain scenarios (simpler implementation and specific handling of learning rates).

Finally, we observed that here, independently of the order, for Spanish and English, this architecture gave us the better results (considering accuracy, which is biased by the amount of words and characters. Eg: if sentence as 4 words and 1 is wrong the accuracy will decrease a lot, however if we have a wrong character as there are more instances it will be less affected). 

	
