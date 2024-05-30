In this file we would explain our hyperparameter exploration for our model.

We did a thorough search of hyperparameters to understand better our model and our data and see which ones maximize the accuracy and other metrics we computed, such as ‘wer’ 
(Word error rate), ‘cer’ (Character error rate) and ‘bleu’ (Bilingual Evaluation Understudy). 

Since, our time, as in, with a virtual machine is limited we couldn’t try all combinations, so we changed our approach to look what parameters got better results and then do slight modifications. We also had a hypothesis in mind, that was, hyperparameters that maximize our metrics should be similar when doing reverse translation (taking the input language as the output and the other way around). Another theory that we wanted to confirm was that if we use a different architecture, but the languages are the same, the hyperparameters would be similar.

In the repository, there’s a folder called ‘images’, we have saved both models, word-based, and character-based. In each folder, we save the plots with the different results for each pair of languages. And then compute different metrics in each set (train, validation, and test), and saving them in their respective folder as well. Each set have their own best architecture for each metric, as shown in the plots, but to have an overall vision, we will talk about the results for ‘Accuracy’, on the train and test set.

For the word-based model we got this results:
On the train set, the best results when translating from English to Spanish: 

	-LSTM with a latent dimension of 256, using a Negative log likelihood Loss, an Adam optimizer, a learning rate of 0.001 and a dropout of 0. It has a 0.91 acuracy. 

 ![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/f1a642b2-897b-4087-ac46-697c0ca00741)

From Spanish to English, however the best was a 

 	-GRU, with a latent dimension of 256, NLLL loss, Adam, lr = 0.001 and dropout 0.2, we got a 0.84 training accuracy.


On the test set, when translating from English to Spanish, we found this configuration to be the best one:		

	-LSTM with a latent dimension of 256, using a Cross Entropy Loss, an Adam optimizer, a learning rate of 0.001 and a dropout of 0.2. The result was a 0.55 acuracy. 

![imagen_2024-05-29_225423790](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/5d7b24d1-5d17-4276-8edf-b45bf4b8404c)

However, it is remarkable, since it got similar results a GRU with 256 latent dim, a negative log likelihood loss (NLLL), adagrad, lr = 0.001 and dropout = 0.

When translating from Spanish to English, we observed that the configuration that was the best one was:

	-LSTM with a latent dimension of 256, using a Cross Entropy Loss, an Adam optimizer, a learning rate of 0.001 and a dropout of 0.2, with a 0.64 accuracy.

Which what we considered at the start. The second-best result was not exactly what we tried, but close enough, it was a GRU with 256 latent dim, a negative log likelihood loss (NLLL), Adam , lr = 0.001 but with a dropout of 0.2 as well.

![imagen_2024-05-29_225510726](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/ad51c1eb-c69c-4edd-9d3b-314f06b377ea)



Then, with the character-based model we saw this configuration to be the most accurate:
On the training set, when doing Eng->Spa

	-LSTM latent dim = 256, NLLL loss, opt = RMSProp, lr of 0.001 and dropout = 0.2, accuracy = 0.78

![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/1b01f293-8be5-4955-b734-a3f0435be31a)

And from Spanish to English:

	-LSTM, latent_dim = 256, NLLL loss, opt = RMSProp, lr = 0.001, dropout = 0.2

![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/f655ebdd-0774-4222-a87c-25182dc0c9af)




And on the test set, from English to Spanish: 

	-LSTM with a latent dimension of 256, a negative log likelihood loss, an RMSProp optimizer, a learning rate of 0.001 and a dropout of 0.2, with that, we managed to get a 0.75 accuracy.

 ![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/e0779305-5e85-43f7-be84-8f43b1041383)


And from Spanish to English:

	-LSTM with a latent dimension of 256, a negative log likelihood loss, an RMSProp optimizer, a learning rate of 0.001 and a dropout of 0.2, this had a 0.78 accuracy.

 ![image](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/ad05787b-18d9-4859-ba78-630d7881f98c)


We observed that here, independently of the order, for Spanish and English, this architecture gave us the better results. 
	
