In this file we would explain our hyperparameter exploration for our model.

We did a thorough search of hyperparameters to understand better our model and our data and see which ones maximize the accuracy and other metrics we computed, such as ‘wer’ 
(Word error rate), ‘cer’ (Character error rate) and ‘bleu’ (Bilingual Evaluation Understudy). 

Since, our time, as in, with a virtual machine is limited we couldn’t try all combinations, so we changed our approach to look what parameters got better results and then do slight modifications. We also had a hypothesis in mind, that was, hyperparameters that maximize our metrics should be similar when doing reverse translation (taking the input language as the output and the other way around). Another theory that we wanted to confirm was that if we use a different architecture, but the languages are the same, the hyperparameters would be similar.

In the repository, there’s a folder called ‘images’, where we save the plots with the different results for each pair of languages. And then compute different metrics in each set (train, validation, and test), and saving them in their respective folder as well. Each set have their own best architecture for each metric, as shown in the plots, but to have an overall vision, we will talk about the results for ‘Accuracy’, on the test set.

When translating from English to Spanish, we found this configuration to be the best one on the test set:		
	-LSTM with a latent dimension of 256, using a Cross Entropy Loss, an Adam optimizer, a learning rate of 0.001 and a dropout of 0.2.

![imagen_2024-05-29_225423790](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/5d7b24d1-5d17-4276-8edf-b45bf4b8404c)

However, it is remarkable, since it got similar results a GRU with 256 latent dim, a negative log likelihood loss (NLLL), adagrad, lr = 0.001 and dropout = 0.

When translating from Spanish to English, we observed that the configuration that was the best one was:

	-LSTM with a latent dimension of 256, using a Cross Entropy Loss, an Adam optimizer, a learning rate of 0.001 and a dropout of 0.2,

Which what we considered at the start. The second-best result was not exactly what we tried, but close enough, it was a GRU with 256 latent dim, a negative log likelihood loss (NLLL), Adam , lr = 0.001 but with a dropout of 0.2 as well.

![imagen_2024-05-29_225510726](https://github.com/DCC-UAB/deep-learning-project-2024-ai_nndl_group_05/assets/93304682/ad51c1eb-c69c-4edd-9d3b-314f06b377ea)
